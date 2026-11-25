"""Silo media server backend, built on Silo's native ``/api/v2`` admin API.

Silo (https://github.com/Silo-Server/silo-server) also serves Jellyfin-shaped
routes, but that compatibility layer only ever acts as a non-administrator
viewer: it cannot list other users' sessions or watch state, and Silo plans to
retire it. Everything here therefore goes through the native admin API with an
administrator's ``sa_`` API key:

* inventory: ``GET /api/v2/admin/catalog/items`` (movies, series, episodes,
  provider ids, every present file with its size and, with
  ``include_file_paths``, its path on the Silo host)
* watch data: ``GET /api/v2/admin/playback-history`` (finalized plays across
  every account and profile)
* live playback: ``GET /api/v2/admin/sessions``
* rescans: ``POST /api/v2/scan``

Silo never deletes media (a permanent non-goal), so removals must go through
Radarr/Sonarr; :meth:`SiloService.delete_item` refuses explicitly.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any

import niquests
from niquests.exceptions import ReadTimeout
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from backend.core.codecs import (
    normalize_audio_codec_family,
    normalize_video_codec_family,
)
from backend.core.logger import LOG
from backend.core.utils.filesystem import normalize_fpath
from backend.core.utils.request import format_http_failure, should_retry_on_status
from backend.enums import MediaType, Service
from backend.models.live_playback import LiveSession, PlaybackSnapshot
from backend.models.media import (
    AggregatedEpisodeData,
    AggregatedMovieData,
    AggregatedSeasonData,
    AggregatedSeriesData,
    EpisodeVersionData,
    ExternalIDs,
    MediaWatchSnapshot,
    MovieVersionData,
)
from backend.models.services.health import HealthResult

JsonDict = dict[str, Any]

API_PREFIX = "api/v2"
# admin/catalog/items and admin/playback-history both cap pages at 200
PAGE_LIMIT = 200
# admin/catalog/items accepts at most this many repeated content_ids
CONTENT_IDS_LIMIT = 200

MOVIE_LIBRARY_TYPES = frozenset({"movies", "mixed"})
SERIES_LIBRARY_TYPES = frozenset({"series", "mixed"})

# Silo's dynamic_range values that mean "some flavour of HDR"
_HDR_RANGES = frozenset({"hdr10", "hdr10_plus", "hlg", "dolby_vision", "hdr_unknown"})


class SiloUnsupportedOperation(RuntimeError):
    """Raised for operations Silo deliberately does not offer."""


def _parse_datetime(raw: object) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_int(raw: object) -> int | None:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return int(str(raw).strip())
    except ValueError:
        return None


def _resolution_height(resolution: str | None) -> int | None:
    """Map Silo's resolution label (``2160p``, ``1080p``...) to a pixel height."""
    if not resolution:
        return None
    text = resolution.strip().lower()
    if text in {"4k", "uhd"}:
        return 2160
    if text.endswith(("p", "i")) and text[:-1].isdigit():
        return int(text[:-1])
    return None


def watch_user_key(username: str | None, profile_name: str | None) -> str | None:
    """Stable per-viewer key for a Silo (account, profile) pair.

    A Silo account holds several household profiles, and each profile is one
    viewer. The account's own name identifies its primary profile (whose name
    usually matches); any other profile is qualified with the account name so
    two households' "Kids" profiles stay distinct.
    """
    user = (username or "").strip()
    profile = (profile_name or "").strip()
    if not user:
        return profile or None
    if not profile or profile.casefold() == user.casefold():
        return user
    return f"{user} ({profile})"


@dataclass(slots=True)
class _PlayStats:
    """Completed plays of one catalog item across every viewer."""

    view_count: int = 0
    last_viewed_at: datetime | None = None
    viewers: set[str] = field(default_factory=set)

    def add(self, ended_at: datetime | None, viewer: str | None) -> None:
        self.view_count += 1
        if ended_at and (self.last_viewed_at is None or ended_at > self.last_viewed_at):
            self.last_viewed_at = ended_at
        if viewer:
            self.viewers.add(viewer)

    def merge(self, other: _PlayStats) -> None:
        self.view_count += other.view_count
        if other.last_viewed_at and (
            self.last_viewed_at is None or other.last_viewed_at > self.last_viewed_at
        ):
            self.last_viewed_at = other.last_viewed_at
        self.viewers |= other.viewers


class SiloService:
    """Silo media server backend (native API, administrator API key)."""

    __slots__ = ("api_key", "service_url", "service_type", "session", "_libraries")

    def __init__(self, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.service_url = base_url.rstrip("/")
        self.service_type = Service.SILO
        self.session = niquests.AsyncSession(timeout=300)
        for name, value in self._headers(api_key).items():
            self.session.headers[name] = value
        # library id -> library row, cached for the client's lifetime
        self._libraries: dict[str, JsonDict] | None = None

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {api_key}", "accept": "application/json"}

    def _url(self, endpoint: str) -> str:
        return f"{self.service_url}/{API_PREFIX}/{endpoint.lstrip('/')}"

    # ------------------------------------------------------------------ http

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=(
            retry_if_exception_type((ConnectionError, TimeoutError, ReadTimeout))
            | retry_if_exception(should_retry_on_status)
        ),
    )
    async def _make_request(
        self,
        endpoint: str,
        params: Iterable[tuple[str, object]] | None = None,
        **kwargs: Any,
    ) -> JsonDict:
        """GET a native API endpoint and return its JSON object.

        ``params`` is a list of pairs so filters such as ``content_ids`` can
        repeat; values are sent as strings.
        """
        query: list[tuple[str, str | list[str] | None]] = [
            (key, str(value)) for key, value in params or ()
        ]
        response = await self.session.get(
            self._url(endpoint), params=query or None, **kwargs
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"Silo {endpoint} returned a non-object response")
        return payload

    async def _paginate(
        self, endpoint: str, params: list[tuple[str, Any]] | None = None
    ) -> AsyncIterator[JsonDict]:
        """Yield every row of a cursor-paged ``{items, page}`` collection."""
        base = list(params or [])
        cursor: str | None = None
        while True:
            page_params = [*base, ("limit", PAGE_LIMIT)]
            if cursor:
                page_params.append(("cursor", cursor))
            payload = await self._make_request(endpoint, params=page_params)
            items = payload.get("items")
            if not isinstance(items, list):
                raise ValueError(f"Silo {endpoint} returned no items list")
            for item in items:
                if isinstance(item, dict):
                    yield item
            page = payload.get("page")
            if not isinstance(page, dict) or not page.get("has_more"):
                return
            cursor = page.get("next_cursor")
            if not cursor:
                return

    # ---------------------------------------------------------------- health

    async def health(self) -> HealthResult:
        """Check the server is reachable and the key holds administrator access."""
        try:
            # system/info answers anonymously, so probe an admin route as well
            await self._make_request("system/info")
            await self._make_request("admin/catalog/capabilities")
        except Exception as exc:
            return HealthResult.failed(
                format_http_failure(action="Silo health check", exception=exc)
            )
        return HealthResult.healthy()

    @staticmethod
    async def test_service(url: str, api_key: str) -> bool:
        """Test a Silo connection and administrator key without full setup."""
        async with niquests.AsyncSession() as session:
            response = await session.get(
                f"{url.rstrip('/')}/{API_PREFIX}/admin/catalog/capabilities",
                headers=SiloService._headers(api_key),
            )
            response.raise_for_status()
            if response.status_code == 200:
                return True
            raise ValueError(f"Unexpected status code: {response.status_code}")

    # ------------------------------------------------------------- libraries

    async def _get_libraries(self) -> dict[str, JsonDict]:
        if self._libraries is None:
            libraries: dict[str, JsonDict] = {}
            async for row in self._paginate("libraries"):
                library_id = str(row.get("id") or "")
                if library_id and row.get("enabled", True):
                    libraries[library_id] = row
            self._libraries = libraries
        return self._libraries

    async def _get_media_libraries(
        self, library_types: frozenset[str]
    ) -> list[dict[str, str]]:
        return [
            {"id": library_id, "name": str(row.get("name") or library_id)}
            for library_id, row in (await self._get_libraries()).items()
            if str(row.get("type") or "").lower() in library_types
        ]

    async def get_movie_libraries(self) -> list[dict[str, str]]:
        """Movie-capable libraries (``movies`` and ``mixed``) as ``{id, name}``."""
        return await self._get_media_libraries(MOVIE_LIBRARY_TYPES)

    async def get_series_libraries(self) -> list[dict[str, str]]:
        """Series-capable libraries (``series`` and ``mixed``) as ``{id, name}``."""
        return await self._get_media_libraries(SERIES_LIBRARY_TYPES)

    async def _selected_library_ids(
        self, library_types: frozenset[str], included_libraries: list[str] | None
    ) -> dict[str, str]:
        """Library id -> name for the libraries a sync should read."""
        libraries = await self._get_media_libraries(library_types)
        if included_libraries:
            wanted = set(included_libraries)
            libraries = [lib for lib in libraries if lib["name"] in wanted]
        return {lib["id"]: lib["name"] for lib in libraries}

    # ------------------------------------------------------------- inventory

    async def _catalog_items(
        self,
        item_type: str,
        *,
        library_id: str | None = None,
        content_ids: Iterable[str] | None = None,
    ) -> AsyncIterator[JsonDict]:
        params: list[tuple[str, Any]] = [
            ("type", item_type),
            ("include_file_paths", "true"),
        ]
        if library_id:
            params.append(("library_id", library_id))
        if content_ids is None:
            async for row in self._paginate("admin/catalog/items", params):
                yield row
            return
        ids = list(dict.fromkeys(content_ids))
        for start in range(0, len(ids), CONTENT_IDS_LIMIT):
            chunk = [
                ("content_ids", cid) for cid in ids[start : start + CONTENT_IDS_LIMIT]
            ]
            async for row in self._paginate("admin/catalog/items", params + chunk):
                yield row

    async def is_inventory_empty(self, media_type: MediaType) -> bool:
        """Confirm emptiness straight from the admin catalog listing."""
        item_type = "movie" if media_type is MediaType.MOVIE else "series"
        payload = await self._make_request(
            "admin/catalog/items", params=[("type", item_type), ("limit", 1)]
        )
        items = payload.get("items")
        if not isinstance(items, list):
            raise ValueError("Silo returned an invalid inventory response")
        return not items

    @staticmethod
    def _external_ids(row: Mapping[str, Any]) -> ExternalIDs | None:
        tmdb = _parse_int(row.get("tmdb_id"))
        if tmdb is None:
            return None
        return ExternalIDs(
            tmdb=tmdb,
            imdb=str(row["imdb_id"]) if row.get("imdb_id") else None,
            tmdb_collection=None,
            tvdb=str(row["tvdb_id"]) if row.get("tvdb_id") else None,
        )

    @staticmethod
    def _version_signals(version: Mapping[str, Any]) -> JsonDict:
        """Fields shared by movie and episode version data."""
        dynamic_range = str(version.get("dynamic_range") or "").lower() or None
        resolution = str(version.get("resolution") or "") or None
        bitrate_kbps = _parse_int(version.get("bitrate_kbps"))
        return {
            "video_resolution": resolution,
            "video_height": _resolution_height(resolution),
            "video_codec_family": normalize_video_codec_family(
                version.get("codec_video")
            ),
            "video_hdr": None
            if dynamic_range is None
            else dynamic_range in _HDR_RANGES,
            "video_dolby_vision": (
                None if dynamic_range is None else dynamic_range == "dolby_vision"
            ),
            # Silo reports the whole file's bitrate; it's the best video estimate
            "video_bitrate": bitrate_kbps * 1000 if bitrate_kbps else None,
            "audio_codec_family": normalize_audio_codec_family(
                version.get("codec_audio")
            ),
            "audio_channels": _parse_int(version.get("audio_channels")),
        }

    def _movie_version(
        self,
        row: Mapping[str, Any],
        version: Mapping[str, Any],
        library_names: Mapping[str, str],
    ) -> MovieVersionData | None:
        library_id = str(version.get("library_id") or "")
        file_id = str(version.get("file_id") or "")
        if not file_id or library_id not in library_names:
            return None
        path = version.get("file_path")
        signals = self._version_signals(version)
        return MovieVersionData(
            service=Service.SILO,
            service_item_id=str(row["content_id"]),
            service_media_id=file_id,
            library_id=library_id,
            library_name=library_names[library_id],
            path=str(path) if path else None,
            size=_parse_int(version.get("file_size")) or 0,
            added_at=_parse_datetime(row.get("added_at")),
            file_name=PurePosixPath(str(path)).name if path else None,
            container=str(version.get("container") or "") or None,
            video_track_count=1 if version.get("codec_video") else None,
            video_codec=str(version.get("codec_video") or "") or None,
            audio_codec=str(version.get("codec_audio") or "") or None,
            **signals,
        )

    # ---------------------------------------------------------------- watch

    async def _completed_plays(self) -> AsyncIterator[JsonDict]:
        async for row in self._paginate(
            "admin/playback-history", [("completed", "true")]
        ):
            yield row

    async def _play_stats_by_item(self) -> tuple[dict[str, _PlayStats], bool]:
        """Completed plays per played content id, and whether the read finished."""
        stats: dict[str, _PlayStats] = defaultdict(_PlayStats)
        try:
            async for row in self._completed_plays():
                item_id = str(row.get("media_item_id") or "")
                if not item_id:
                    continue
                stats[item_id].add(
                    _parse_datetime(row.get("ended_at")),
                    watch_user_key(row.get("username"), row.get("profile_name")),
                )
        except Exception as exc:
            LOG.warning(
                format_http_failure(
                    action="Silo playback history read incomplete", exception=exc
                )
            )
            return dict(stats), False
        return dict(stats), True

    async def get_aggregated_movies(
        self, included_libraries: list[str] | None = None
    ) -> list[AggregatedMovieData]:
        """Movies with every present file and completed plays across viewers."""
        library_names = await self._selected_library_ids(
            MOVIE_LIBRARY_TYPES, included_libraries
        )
        if not library_names:
            return []
        rows: dict[str, JsonDict] = {}
        for library_id in library_names:
            async for row in self._catalog_items("movie", library_id=library_id):
                rows.setdefault(str(row.get("content_id")), row)

        plays, plays_complete = await self._play_stats_by_item()
        movies: list[AggregatedMovieData] = []
        for content_id, row in rows.items():
            external_ids = self._external_ids(row)
            if external_ids is None:
                LOG.debug(f"Skipping Silo movie {content_id}: no TMDB id")
                continue
            versions = [
                version
                for raw in row.get("versions") or []
                if isinstance(raw, dict)
                and (version := self._movie_version(row, raw, library_names))
            ]
            if not versions:
                continue
            stats = plays.get(content_id, _PlayStats())
            movies.append(
                AggregatedMovieData(
                    name=str(row.get("title") or ""),
                    year=_parse_int(row.get("year")),
                    external_ids=external_ids,
                    versions=versions,
                    view_count=stats.view_count,
                    last_viewed_at=stats.last_viewed_at,
                    played_by_user_count=len(stats.viewers),
                    media_server_user_rating=None,
                    watch_history_complete=plays_complete,
                )
            )
        return movies

    async def get_aggregated_series(
        self, included_libraries: list[str] | None = None
    ) -> list[AggregatedSeriesData]:
        """Series with season/episode breakdown and completed plays."""
        library_names = await self._selected_library_ids(
            SERIES_LIBRARY_TYPES, included_libraries
        )
        if not library_names:
            return []
        series_rows: dict[str, JsonDict] = {}
        episodes_by_series: dict[str, dict[str, JsonDict]] = defaultdict(dict)
        for library_id in library_names:
            async for row in self._catalog_items("series", library_id=library_id):
                series_rows.setdefault(str(row.get("content_id")), row)
            async for row in self._catalog_items("episode", library_id=library_id):
                series_id = str(row.get("series_id") or "")
                if series_id:
                    episodes_by_series[series_id].setdefault(
                        str(row.get("content_id")), row
                    )

        plays, plays_complete = await self._play_stats_by_item()
        results: list[AggregatedSeriesData] = []
        for series_id, row in series_rows.items():
            external_ids = self._external_ids(row)
            if external_ids is None:
                LOG.debug(f"Skipping Silo series {series_id}: no TMDB id")
                continue
            library_id = next(
                (lid for lid in row.get("library_ids") or [] if lid in library_names),
                None,
            )
            if library_id is None:
                continue
            seasons = self._build_seasons(
                series_id,
                episodes_by_series.get(series_id, {}).values(),
                plays,
                library_names,
            )
            if not seasons:
                continue
            series_stats = _PlayStats()
            for episode_id in episodes_by_series.get(series_id, {}):
                if episode_id in plays:
                    series_stats.merge(plays[episode_id])
            season_paths = [s.path for s in seasons if s.path]
            results.append(
                AggregatedSeriesData(
                    id=series_id,
                    name=str(row.get("title") or ""),
                    year=_parse_int(row.get("year")),
                    service=Service.SILO,
                    library_name=library_names[library_id],
                    library_id=library_id,
                    path=_series_path(season_paths),
                    added_at=_parse_datetime(row.get("added_at")),
                    external_ids=external_ids,
                    size=sum(s.size for s in seasons),
                    view_count=series_stats.view_count,
                    last_viewed_at=series_stats.last_viewed_at,
                    played_by_user_count=len(series_stats.viewers),
                    media_server_genres=[str(g) for g in row.get("genres") or []]
                    or None,
                    season_data=seasons,
                    watch_history_complete=plays_complete,
                )
            )
        return results

    def _build_seasons(
        self,
        series_id: str,
        episodes: Iterable[JsonDict],
        plays: Mapping[str, _PlayStats],
        library_names: Mapping[str, str],
    ) -> list[AggregatedSeasonData]:
        by_season: dict[int, list[AggregatedEpisodeData]] = defaultdict(list)
        for row in episodes:
            season_number = _parse_int(row.get("season_number"))
            episode_number = _parse_int(row.get("episode_number"))
            if season_number is None or episode_number is None:
                continue
            content_id = str(row.get("content_id"))
            versions = tuple(
                EpisodeVersionData(
                    service=Service.SILO,
                    service_item_id=content_id,
                    service_media_id=str(v["file_id"]),
                    library_id=str(v["library_id"]),
                    library_name=library_names[str(v["library_id"])],
                    path=str(v["file_path"]) if v.get("file_path") else None,
                    size=_parse_int(v.get("file_size")) or 0,
                    added_at=_parse_datetime(row.get("added_at")),
                    **self._version_signals(v),
                )
                for v in row.get("versions") or []
                if isinstance(v, dict)
                and v.get("file_id")
                and str(v.get("library_id")) in library_names
            )
            if not versions:
                continue
            stats = plays.get(content_id, _PlayStats())
            by_season[season_number].append(
                AggregatedEpisodeData(
                    episode_number=episode_number,
                    view_count=stats.view_count,
                    name=str(row.get("title") or "") or None,
                    air_date=_parse_datetime(row.get("air_date")),
                    last_viewed_at=stats.last_viewed_at,
                    added_at=_parse_datetime(row.get("added_at")),
                    size=sum(v.size for v in versions),
                    path=versions[0].path,
                    runtime_seconds=None,
                    versions=versions,
                )
            )

        seasons: list[AggregatedSeasonData] = []
        for season_number, season_episodes in sorted(by_season.items()):
            season_episodes.sort(key=lambda e: e.episode_number)
            episode_paths = [e.path for e in season_episodes if e.path]
            last_viewed = max(
                (e.last_viewed_at for e in season_episodes if e.last_viewed_at),
                default=None,
            )
            added = [e.added_at for e in season_episodes if e.added_at]
            aired = [e.air_date for e in season_episodes if e.air_date]
            heights = [
                v.video_height
                for e in season_episodes
                for v in e.versions
                if v.video_height
            ]
            hdr_flags = [
                v.video_hdr
                for e in season_episodes
                for v in e.versions
                if v.video_hdr is not None
            ]
            seasons.append(
                AggregatedSeasonData(
                    service_series_id=series_id,
                    season_number=season_number,
                    size=sum(e.size or 0 for e in season_episodes),
                    episode_count=len(season_episodes),
                    view_count=sum(e.view_count for e in season_episodes),
                    last_viewed_at=last_viewed,
                    added_at=min(added) if added else None,
                    air_date=min(aired) if aired else None,
                    service_season_id=None,
                    path=_common_directory(episode_paths),
                    episode_paths=episode_paths or None,
                    episode_data=season_episodes,
                    has_hdr=any(hdr_flags) if hdr_flags else None,
                    max_video_height=max(heights) if heights else None,
                )
            )
        return seasons

    async def get_watched_user_snapshots(self) -> list[MediaWatchSnapshot]:
        """Latest completed play per viewer for every movie and episode.

        Only real plays are visible to an administrator: Silo keeps titles a
        viewer *marked* watched in profile-scoped state that the admin API
        does not expose.
        """
        latest: dict[str, dict[str, tuple[datetime, int]]] = defaultdict(dict)
        async for row in self._completed_plays():
            item_id = str(row.get("media_item_id") or "")
            viewer = watch_user_key(row.get("username"), row.get("profile_name"))
            ended_at = _parse_datetime(row.get("ended_at"))
            if not item_id or not viewer or ended_at is None:
                continue
            previous = latest[item_id].get(viewer)
            count = (previous[1] if previous else 0) + 1
            newest = max(ended_at, previous[0]) if previous else ended_at
            latest[item_id][viewer] = (newest, count)
        if not latest:
            return []

        snapshots: list[MediaWatchSnapshot] = []
        series_tmdb: dict[str, int | None] = {}
        episodes: list[JsonDict] = []
        for media_type in ("movie", "episode"):
            async for row in self._catalog_items(media_type, content_ids=latest):
                if media_type == "movie":
                    tmdb = _parse_int(row.get("tmdb_id"))
                    if tmdb is None:
                        continue
                    for viewer, (when, count) in latest[str(row["content_id"])].items():
                        snapshots.append(
                            MediaWatchSnapshot(
                                media_type=MediaType.MOVIE,
                                tmdb_id=tmdb,
                                watch_user_key=viewer,
                                last_watched_at=when,
                                play_count=count,
                                source_item_id=str(row["content_id"]),
                            )
                        )
                else:
                    episodes.append(row)
                    if row.get("series_id"):
                        series_tmdb.setdefault(str(row["series_id"]), None)

        if series_tmdb:
            async for row in self._catalog_items("series", content_ids=series_tmdb):
                series_tmdb[str(row["content_id"])] = _parse_int(row.get("tmdb_id"))
        for row in episodes:
            tmdb = series_tmdb.get(str(row.get("series_id") or ""))
            if tmdb is None:
                continue
            for viewer, (when, count) in latest[str(row["content_id"])].items():
                snapshots.append(
                    MediaWatchSnapshot(
                        media_type=MediaType.SERIES,
                        tmdb_id=tmdb,
                        watch_user_key=viewer,
                        last_watched_at=when,
                        play_count=count,
                        source_item_id=str(row["content_id"]),
                        season_number=_parse_int(row.get("season_number")),
                        episode_number=_parse_int(row.get("episode_number")),
                    )
                )
        return snapshots

    # --------------------------------------------------------- live playback

    async def get_live_sessions(self) -> PlaybackSnapshot:
        """Live plays across every account, with each played file's path."""
        rows: list[JsonDict] = []
        try:
            async for row in self._paginate("admin/sessions"):
                rows.append(row)
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            http = f"; HTTP {status}" if isinstance(status, int) else ""
            return PlaybackSnapshot(
                available=False,
                unavailable_reason=f"Silo session read failed; {type(exc).__name__}{http}",
            )
        if not rows:
            return PlaybackSnapshot()

        item_ids = {str(r["media_item_id"]) for r in rows if r.get("media_item_id")}
        paths_by_file: dict[str, str] = {}
        series_by_item: dict[str, str] = {}
        unavailable_reason: str | None = None
        try:
            for media_type in ("movie", "episode"):
                async for item in self._catalog_items(media_type, content_ids=item_ids):
                    if item.get("series_id"):
                        series_by_item[str(item["content_id"])] = str(item["series_id"])
                    for version in item.get("versions") or []:
                        if isinstance(version, dict) and version.get("file_path"):
                            paths_by_file[str(version.get("file_id"))] = str(
                                version["file_path"]
                            )
        except Exception as exc:
            unavailable_reason = f"session path lookup failed; {type(exc).__name__}"

        sessions: list[LiveSession] = []
        for row in rows:
            item_id = str(row.get("media_item_id") or "") or None
            file_id = str(row.get("media_file_id") or "") or None
            path = paths_by_file.get(file_id or "")
            if item_id is None:
                unavailable_reason = "session has no media identity"
            elif path is None and unavailable_reason is None:
                unavailable_reason = "session path lookup returned no paths"
            parent = series_by_item.get(item_id or "")
            sessions.append(
                LiveSession(
                    item_id=item_id,
                    media_id=file_id,
                    paths=(path,) if path else (),
                    parent_ids=(parent,) if parent else (),
                )
            )
        return PlaybackSnapshot(
            sessions=tuple(sessions),
            available=unavailable_reason is None,
            unavailable_reason=unavailable_reason,
        )

    # ------------------------------------------------------------ mutations

    async def scan_item_path(
        self, item_path: str, *, warn_unmatched: bool = True
    ) -> bool:
        """Ask Silo to rescan the library root holding ``item_path``."""
        try:
            scan_path = normalize_fpath(item_path).rstrip("/")
            if not scan_path:
                return False
            for library_id, library in (await self._get_libraries()).items():
                for root in library.get("paths") or []:
                    root_path = normalize_fpath(str(root)).rstrip("/")
                    if root_path and (
                        scan_path == root_path or scan_path.startswith(root_path + "/")
                    ):
                        response = await self.session.post(
                            self._url("scan"),
                            json={"library_id": library_id, "path": item_path},
                        )
                        response.raise_for_status()
                        LOG.debug(
                            f"Triggered Silo scan for {item_path} in library {library_id}"
                        )
                        return True
            log = LOG.warning if warn_unmatched else LOG.debug
            log(f"Could not find Silo library for path: {item_path}")
            return False
        except Exception as exc:
            LOG.error(
                format_http_failure(
                    action=f"Failed to scan Silo path {item_path}", exception=exc
                )
            )
            return False

    async def delete_item(self, item_id: str) -> None:
        """Silo never deletes media; removals must go through Radarr/Sonarr."""
        raise SiloUnsupportedOperation(
            f"Silo does not delete media (item {item_id}); "
            "delete through Radarr/Sonarr instead"
        )

    async def delete_movie_version(self, item_id: str, _media_source_id: str) -> None:
        """Silo never deletes media; removals must go through Radarr."""
        await self.delete_item(item_id)

    async def get_series_episode_order(
        self, item_id: str, *, library_orders: dict[str, str | None] | None = None
    ) -> str | None:
        """Silo numbers episodes by their aired order only."""
        return "aired"


def _common_directory(paths: list[str]) -> str | None:
    """Deepest directory holding every path (the season folder)."""
    parents = [PurePosixPath(normalize_fpath(p)).parent for p in paths if p]
    if not parents:
        return None
    common = parents[0]
    for parent in parents[1:]:
        while common != parent and common not in parent.parents:
            if common == common.parent:
                return None
            common = common.parent
    return str(common)


def _series_path(season_paths: list[str]) -> str | None:
    """Series folder: the common parent of seasons, or a single season's parent.

    With one season the series folder can't be told apart from the season's
    own folder, so assume the conventional ``Show/Season N`` layout only when
    the folder name says it's a season.
    """
    if not season_paths:
        return None
    if len(set(season_paths)) == 1:
        only = PurePosixPath(season_paths[0])
        name = only.name.lower()
        is_season_folder = name.startswith(("season", "specials")) or name in {
            "s0",
            "s00",
        }
        return str(only.parent) if is_season_folder else str(only)
    return _common_directory([f"{p}/x" for p in season_paths])
