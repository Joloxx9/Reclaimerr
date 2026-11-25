from typing import Literal

from backend.enums import Service

MediaServerType = Literal[Service.JELLYFIN, Service.EMBY, Service.PLEX, Service.SILO]
# Silo joins MEDIA_SERVERS once service_manager can initialize it.
MEDIA_SERVERS = frozenset[Service]({Service.JELLYFIN, Service.EMBY, Service.PLEX})
