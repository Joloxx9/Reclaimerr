"""SiloService against a fake of Silo's native /api/v2 admin routes."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

from backend.enums import MediaType, Service
from backend.services.silo import (
    SiloService,
    SiloUnsupportedOperation,
    _series_path,
    watch_user_key,
)

LIBRARIES = [
    {
        "id": "1",
        "name": "Movies",
        "type": "movies",
        "enabled": True,
        "paths": ["/media/movies"],
    },
    {
        "id": "2",
        "name": "TV",
        "type": "series",
        "enabled": True,
        "paths": ["/media/tv"],
    },
    {
        "id": "3",
        "name": "Books",
        "type": "ebooks",
        "enabled": True,
        "paths": ["/media/books"],
    },
    {
        "id": "4",
        "name": "Old",
        "type": "movies",
        "enabled": False,
        "paths": ["/media/old"],
    },
]


def version(file_id: str, library_id: str, path: str, size: int, **extra: Any):
    return {
        "file_id": file_id,
        "library_id": library_id,
        "file_path": path,
        "file_size": size,
        **extra,
    }


CATALOG = [
    {
        "content_id": "movie-tmdb-603",
        "type": "movie",
        "title": "The Matrix",
        "year": 1999,
        "tmdb_id": "603",
        "imdb_id": "tt0133093",
        "library_ids": ["1"],
        "added_at": "2026-01-02T03:04:05Z",
        "versions": [
            version(
                "10",
                "1",
                "/media/movies/The Matrix (1999)/The Matrix.mkv",
                4_000,
                resolution="2160p",
                codec_video="hevc",
                codec_audio="truehd",
                audio_channels=8,
                dynamic_range="dolby_vision",
                bitrate_kbps=50_000,
                container="mkv",
            ),
            version(
                "11", "1", "/media/movies/The Matrix (1999)/The Matrix 1080p.mkv", 1_000
            ),
        ],
    },
    {
        "content_id": "movie-no-tmdb",
        "type": "movie",
        "title": "Home Video",
        "library_ids": ["1"],
        "added_at": "2026-01-02T03:04:05Z",
        "versions": [version("12", "1", "/media/movies/home.mkv", 5)],
    },
    {
        "content_id": "series-tmdb-1399",
        "type": "series",
        "title": "Show",
        "year": 2011,
        "tmdb_id": "1399",
        "tvdb_id": "121361",
        "genres": ["Drama"],
        "library_ids": ["2"],
        "added_at": "2026-02-01T00:00:00Z",
        "versions": [],
    },
    *[
        {
            "content_id": f"ep-{s}-{e}",
            "type": "episode",
            "title": f"Episode {e}",
            "series_id": "series-tmdb-1399",
            "season_number": s,
            "episode_number": e,
            "air_date": f"201{s}-04-0{e}",
            "library_ids": ["2"],
            "added_at": "2026-02-01T00:00:00Z",
            "versions": [
                version(
                    f"2{s}{e}",
                    "2",
                    f"/media/tv/Show/Season {s}/S0{s}E0{e}.mkv",
                    100,
                    resolution="1080p",
                    dynamic_range="sdr",
                )
            ],
        }
        for s in (1, 2)
        for e in (1, 2)
    ],
]

HISTORY = [
    {
        "media_item_id": "movie-tmdb-603",
        "username": "anna",
        "profile_name": "anna",
        "ended_at": "2026-03-01T20:00:00Z",
        "completed": True,
    },
    {
        "media_item_id": "movie-tmdb-603",
        "username": "anna",
        "profile_name": "Kids",
        "ended_at": "2026-03-05T20:00:00Z",
        "completed": True,
    },
    {
        "media_item_id": "movie-tmdb-603",
        "username": "anna",
        "profile_name": "anna",
        "ended_at": "2026-02-01T20:00:00Z",
        "completed": True,
    },
    {
        "media_item_id": "ep-1-2",
        "username": "bob",
        "profile_name": "Bob",
        "ended_at": "2026-03-10T21:00:00Z",
        "completed": True,
    },
]


class FakeSilo:
    """Routes SiloService._make_request calls, paging two rows at a time."""

    page_size = 2

    def __init__(
        self, *, sessions: list[dict] | None = None, fail_history: bool = False
    ):
        self.sessions = sessions or []
        self.fail_history = fail_history
        self.calls: list[tuple[str, list[tuple[str, Any]]]] = []

    def rows(self, endpoint: str, params: list[tuple[str, Any]]) -> list[dict]:
        p = dict(params)
        if endpoint == "libraries":
            return LIBRARIES
        if endpoint == "admin/sessions":
            return self.sessions
        if endpoint == "admin/playback-history":
            if self.fail_history:
                raise ConnectionError("history down")
            return HISTORY
        if endpoint == "admin/catalog/items":
            rows = [r for r in CATALOG if r["type"] == p.get("type")]
            if "library_id" in p:
                rows = [r for r in rows if p["library_id"] in r["library_ids"]]
            ids = [v for k, v in params if k == "content_ids"]
            if ids:
                rows = [r for r in rows if r["content_id"] in ids]
            return rows
        raise AssertionError(f"unexpected endpoint {endpoint}")

    async def __call__(self, endpoint: str, params=None, **_: Any) -> dict:
        params = list(params.items() if isinstance(params, dict) else params or [])
        self.calls.append((endpoint, params))
        if endpoint in {"system/info", "admin/catalog/capabilities"}:
            return {}
        rows = self.rows(endpoint, params)
        p = dict(params)
        start = int(p.get("cursor", 0))
        limit = min(int(p.get("limit", 50)), self.page_size)
        page = rows[start : start + limit]
        more = start + limit < len(rows)
        return {
            "items": page,
            "page": {
                "has_more": more,
                "next_cursor": str(start + limit) if more else None,
            },
        }


class PatchableSilo(SiloService):
    """No __slots__, so tests can swap the HTTP layer per instance."""


def run_with(fake: FakeSilo, coro_factory):
    async def run():
        client = PatchableSilo("sa_synthetic", "http://silo.invalid/")
        client._make_request = fake  # type: ignore[method-assign]
        try:
            return await coro_factory(client)
        finally:
            await client.session.close()

    return asyncio.run(run())


def test_watch_user_key_distinguishes_profiles():
    assert watch_user_key("anna", "anna") == "anna"
    assert watch_user_key("anna", "Anna") == "anna"
    assert watch_user_key("anna", "") == "anna"
    assert watch_user_key("anna", "Kids") == "anna (Kids)"
    assert watch_user_key("", "Kids") == "Kids"
    assert watch_user_key(None, None) is None


def test_client_uses_bearer_key_and_native_prefix():
    async def check(client: SiloService):
        assert client.session.headers["Authorization"] == "Bearer sa_synthetic"
        assert (
            client._url("admin/sessions") == "http://silo.invalid/api/v2/admin/sessions"
        )

    run_with(FakeSilo(), check)


def test_libraries_filter_by_kind_and_enabled():
    movies, series = run_with(
        FakeSilo(),
        lambda c: asyncio.gather(c.get_movie_libraries(), c.get_series_libraries()),
    )
    assert movies == [{"id": "1", "name": "Movies"}]
    assert series == [{"id": "2", "name": "TV"}]


def test_aggregated_movies_carry_versions_and_cross_profile_plays():
    movies = run_with(FakeSilo(), lambda c: c.get_aggregated_movies())
    assert [m.name for m in movies] == ["The Matrix"]  # no-TMDB item skipped
    matrix = movies[0]
    assert matrix.external_ids.tmdb == 603
    assert matrix.external_ids.imdb == "tt0133093"
    assert matrix.view_count == 3
    assert matrix.last_viewed_at == datetime(2026, 3, 5, 20, tzinfo=UTC)
    assert matrix.played_by_user_count == 2  # anna + anna (Kids)
    assert matrix.watch_history_complete is True

    uhd, hd = matrix.versions
    assert uhd.service is Service.SILO
    assert (uhd.service_item_id, uhd.service_media_id) == ("movie-tmdb-603", "10")
    assert uhd.library_name == "Movies"
    assert uhd.path == "/media/movies/The Matrix (1999)/The Matrix.mkv"
    assert uhd.file_name == "The Matrix.mkv"
    assert uhd.size == 4_000
    assert uhd.video_height == 2160
    assert uhd.video_hdr is True and uhd.video_dolby_vision is True
    assert uhd.video_bitrate == 50_000_000
    assert uhd.audio_channels == 8
    # unprobed file: signals stay unknown rather than false
    assert hd.video_hdr is None and hd.video_height is None


def test_included_libraries_filters_by_name():
    assert run_with(FakeSilo(), lambda c: c.get_aggregated_movies(["TV"])) == []


def test_history_failure_marks_watch_data_incomplete():
    movies = run_with(FakeSilo(fail_history=True), lambda c: c.get_aggregated_movies())
    assert movies[0].watch_history_complete is False


def test_aggregated_series_builds_seasons_and_paths():
    (show,) = run_with(FakeSilo(), lambda c: c.get_aggregated_series())
    assert show.id == "series-tmdb-1399"
    assert show.service is Service.SILO
    assert (show.library_id, show.library_name) == ("2", "TV")
    assert show.external_ids.tvdb == "121361"
    assert show.path == "/media/tv/Show"
    assert show.size == 400
    assert show.view_count == 1
    assert show.played_by_user_count == 1
    assert show.media_server_genres == ["Drama"]

    s1, s2 = show.season_data
    assert (s1.season_number, s1.episode_count, s1.size) == (1, 2, 200)
    assert s1.path == "/media/tv/Show/Season 1"
    assert s1.view_count == 1
    assert s1.last_viewed_at == datetime(2026, 3, 10, 21, tzinfo=UTC)
    assert s1.max_video_height == 1080 and s1.has_hdr is False
    assert [e.episode_number for e in s1.episode_data] == [1, 2]
    assert s1.episode_data[1].versions[0].service_item_id == "ep-1-2"
    assert s2.view_count == 0


def test_series_path_for_single_season():
    assert _series_path(["/tv/Show/Season 1"]) == "/tv/Show"
    assert _series_path(["/tv/Miniseries"]) == "/tv/Miniseries"
    assert _series_path([]) is None


def test_watched_user_snapshots_map_to_tmdb_per_viewer():
    snapshots = run_with(FakeSilo(), lambda c: c.get_watched_user_snapshots())
    by_key = {(s.media_type, s.watch_user_key, s.source_item_id): s for s in snapshots}
    anna = by_key[(MediaType.MOVIE, "anna", "movie-tmdb-603")]
    assert anna.tmdb_id == 603
    assert anna.play_count == 2
    assert anna.last_watched_at == datetime(2026, 3, 1, 20, tzinfo=UTC)
    assert by_key[(MediaType.MOVIE, "anna (Kids)", "movie-tmdb-603")].play_count == 1
    bob = by_key[(MediaType.SERIES, "bob", "ep-1-2")]
    assert (bob.tmdb_id, bob.season_number, bob.episode_number) == (1399, 1, 2)
    assert len(snapshots) == 3


def test_live_sessions_resolve_file_paths_and_series():
    fake = FakeSilo(
        sessions=[
            {"media_item_id": "movie-tmdb-603", "media_file_id": "11"},
            {"media_item_id": "ep-2-1", "media_file_id": "221"},
        ]
    )
    snapshot = run_with(fake, lambda c: c.get_live_sessions())
    assert snapshot.available is True
    movie, episode = snapshot.sessions
    assert movie.paths == ("/media/movies/The Matrix (1999)/The Matrix 1080p.mkv",)
    assert movie.parent_ids == ()
    assert episode.paths == ("/media/tv/Show/Season 2/S02E01.mkv",)
    assert episode.parent_ids == ("series-tmdb-1399",)


def test_live_session_without_known_file_makes_snapshot_unavailable():
    fake = FakeSilo(sessions=[{"media_item_id": "gone", "media_file_id": "999"}])
    snapshot = run_with(fake, lambda c: c.get_live_sessions())
    assert snapshot.available is False
    assert snapshot.unavailable_reason == "session path lookup returned no paths"


def test_no_live_sessions():
    snapshot = run_with(FakeSilo(), lambda c: c.get_live_sessions())
    assert snapshot.available is True and snapshot.sessions == ()


def test_inventory_empty_checks_catalog():
    empty = run_with(FakeSilo(), lambda c: c.is_inventory_empty(MediaType.MOVIE))
    assert empty is False


def test_delete_is_refused():
    with pytest.raises(SiloUnsupportedOperation, match="Radarr/Sonarr"):
        run_with(FakeSilo(), lambda c: c.delete_item("movie-tmdb-603"))


def test_scan_item_path_targets_owning_library(monkeypatch):
    posted: list[tuple[str, dict]] = []

    class Response:
        def raise_for_status(self):
            return None

    async def check(client: SiloService):
        async def post(url, json):
            posted.append((url, json))
            return Response()

        monkeypatch.setattr(client.session, "post", post)
        assert await client.scan_item_path("/media/tv/Show/Season 1") is True
        assert (
            await client.scan_item_path("/elsewhere/x", warn_unmatched=False) is False
        )

    run_with(FakeSilo(), check)
    assert posted == [
        (
            "http://silo.invalid/api/v2/scan",
            {"library_id": "2", "path": "/media/tv/Show/Season 1"},
        )
    ]
