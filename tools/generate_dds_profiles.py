#!/usr/bin/env python3
"""Render the P1 rootless-boundary Fast DDS profiles (two distinct roles).

Topology: exactly ONE listening TCP endpoint across the rootless boundary —
the host listener on 17650. Container participants are client-only TCPv4
participants (no listening port at all) that discover the host listener
through an explicit ``initialPeersList`` locator at host-address:17650. Many
independent container OS processes (move_group, ros2_control_node, probes,
...) can therefore share the container role without any TCP bind conflicts:
the previous fixed container listening port is gone.

Roles
-----
host-listener   single Fast DDS participant owning TCP 17650 (listening_ports
                + wan_addr); it accepts clients and needs no initialPeersList.
container-client client-only TCPv4 transport (no server block), explicit
                initialPeersList to host-address:17650. The base
                ``fastdds-container.xml`` leaves the Fast DDS participant ID
                automatic; ``fastdds-container-p<N>.xml`` pins distinct
                participant IDs (1..N) so per-process runs are deterministic
                and provably unique.

``--container-port`` is still accepted (the P0.5 smoke calls it) but is
ignored and deprecated: container participants no longer listen anywhere.

P1.B addition (opt-in via ``--container-mesh``): the generator then also emits
``fastdds-container-mesh.xml`` — still a client-only TCP participant towards
the host listener, plus a container-netns-local UDPv4 transport and the domain
SPDP multicast locator as a second initial peer. SIMPLE discovery over TCP
peers a client only with its initial peer, so the several container processes
of the P1.B MoveIt stack cannot see each other through the P1.A profile alone
(live-evidenced); the local mesh gives them a rendezvous that never crosses
the rootless boundary. Without the flag the output is byte-identical to the
P1.A gate profiles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
from pathlib import Path

TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "config/dds/fastdds-rootless-tcp.xml.in"
TEMPLATE_MESH_PATH = (
    Path(__file__).resolve().parents[1] / "config/dds/fastdds-container-mesh.xml.in"
)

REPORT_SCHEMA = "rocobrick-dds-profiles/v2"
TOPOLOGY = "single-host-listener-17650-client-only-containers"

HOST_PROFILE_NAME = "rocobrick_host_listener"
CONTAINER_PROFILE_NAME = "rocobrick_container_client"
CONTAINER_MESH_PROFILE_NAME = "rocobrick_container_mesh"

# ROS 2 SPDP discovery multicast group and the well-known metatraffic
# multicast port for a domain: PB(7400) + DG(250) * domain_id + d0(0).
SPDP_MULTICAST_GROUP = "239.255.0.1"
SPDP_PORT_BASE = 7400
SPDP_DOMAIN_STEP = 250


def host_address() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("1.1.1.1", 53))
        return str(sock.getsockname()[0])
    finally:
        sock.close()


def render(
    template: str,
    *,
    profile_name: str,
    transport_server: str,
    participant_id: str,
    initial_peer: str,
) -> str:
    """Fill the four role markers; refuse to emit any unresolved marker."""
    result = template.replace("__PROFILE_NAME__", profile_name)
    result = result.replace("__TRANSPORT_SERVER__\n", transport_server)
    result = result.replace("__PARTICIPANT_ID__\n", participant_id)
    result = result.replace("__INITIAL_PEER__\n", initial_peer)
    if "__" in result:
        raise ValueError("unresolved DDS profile marker")
    return result


def host_profile_xml(template: str, *, address: str, host_port: int) -> str:
    transport_server = (
        "      <listening_ports><port>" + str(host_port) + "</port></listening_ports>\n"
        "      <wan_addr>" + address + "</wan_addr>\n"
    )
    return render(
        template,
        profile_name=HOST_PROFILE_NAME,
        transport_server=transport_server,
        participant_id="      <participantID>0</participantID>\n",
        initial_peer="",
    )


def container_peer_xml(address: str, host_port: int) -> str:
    return (
        "        <initialPeersList>\n"
        "          <locator><tcpv4>\n"
        "            <address>" + address + "</address>\n"
        "            <physical_port>" + str(host_port) + "</physical_port>\n"
        "          </tcpv4></locator>\n"
        "        </initialPeersList>\n"
    )


def container_profile_xml(
    template: str, *, address: str, host_port: int, participant_id: int | None
) -> str:
    participant_line = (
        "      <participantID>" + str(participant_id) + "</participantID>\n"
        if participant_id is not None
        else ""
    )
    return render(
        template,
        profile_name=(
            CONTAINER_PROFILE_NAME if participant_id is None
            else CONTAINER_PROFILE_NAME + "_p" + str(participant_id)
        ),
        transport_server="",
        participant_id=participant_line,
        initial_peer=container_peer_xml(address, host_port),
    )


def container_mesh_peer_xml(address: str, host_port: int, multicast_port: int) -> str:
    """Render the mesh initial peers: the host listener + local SPDP group."""
    return (
        "        <initialPeersList>\n"
        "          <locator><tcpv4>\n"
        "            <address>" + address + "</address>\n"
        "            <physical_port>" + str(host_port) + "</physical_port>\n"
        "          </tcpv4></locator>\n"
        "          <locator><udpv4>\n"
        "            <address>" + SPDP_MULTICAST_GROUP + "</address>\n"
        "            <port>" + str(multicast_port) + "</port>\n"
        "          </udpv4></locator>\n"
        "        </initialPeersList>\n"
    )


def container_mesh_profile_xml(
    template: str, *, address: str, host_port: int, multicast_port: int
) -> str:
    """Render the P1.B container mesh profile (client-only TCP + local UDP)."""
    return render(
        template,
        profile_name=CONTAINER_MESH_PROFILE_NAME,
        transport_server="",
        participant_id="",
        initial_peer=container_mesh_peer_xml(address, host_port, multicast_port),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host-address", default=None)
    parser.add_argument("--host-port", type=int, default=17650)
    parser.add_argument(
        "--container-port",
        type=int,
        default=None,
        help="deprecated and ignored: container participants are client-only",
    )
    parser.add_argument(
        "--container-count",
        type=int,
        default=4,
        help="number of pinned-participant-ID container profiles to emit",
    )
    parser.add_argument(
        "--container-mesh",
        action="store_true",
        help="also emit the P1.B container mesh profile (client-only TCP to "
             "the host listener plus a netns-local UDPv4 SPDP rendezvous)",
    )
    parser.add_argument(
        "--ros-domain-id",
        type=int,
        default=0,
        help="ROS domain id (computes the SPDP multicast port for --container-mesh)",
    )
    args = parser.parse_args()

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    mesh_template = TEMPLATE_MESH_PATH.read_text(encoding="utf-8")
    address = args.host_address or host_address()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spdp_multicast_port = SPDP_PORT_BASE + SPDP_DOMAIN_STEP * args.ros_domain_id

    entries: dict[str, dict] = {}

    def write(name: str, text: str, *, role: str, participant_id: int | None,
              listening_ports: list[int], initial_peers: list[list[int]]) -> None:
        path = args.output_dir / f"fastdds-{name}.xml"
        path.write_text(text, encoding="utf-8")
        entries[name] = {
            "role": role,
            "participant_id": participant_id,
            "listening_ports": listening_ports,
            "initial_peers": initial_peers,
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        }

    write(
        "host",
        host_profile_xml(template, address=address, host_port=args.host_port),
        role="host-listener",
        participant_id=0,
        listening_ports=[args.host_port],
        initial_peers=[],
    )
    write(
        "container",
        container_profile_xml(template, address=address, host_port=args.host_port,
                              participant_id=None),
        role="container-client",
        participant_id=None,
        listening_ports=[],
        initial_peers=[[address, args.host_port]],
    )
    for index in range(1, args.container_count + 1):
        write(
            f"container-p{index}",
            container_profile_xml(template, address=address, host_port=args.host_port,
                                  participant_id=index),
            role="container-client",
            participant_id=index,
            listening_ports=[],
            initial_peers=[[address, args.host_port]],
        )

    if args.container_mesh:
        mesh_text = container_mesh_profile_xml(
            mesh_template, address=address, host_port=args.host_port,
            multicast_port=spdp_multicast_port,
        )
        mesh_path = args.output_dir / "fastdds-container-mesh.xml"
        mesh_path.write_text(mesh_text, encoding="utf-8")
        entries["container-mesh"] = {
            "role": "container-client-mesh",
            "participant_id": None,
            "listening_ports": [],
            "initial_peers": [[address, args.host_port]],
            "local_spdp_multicast": [SPDP_MULTICAST_GROUP, spdp_multicast_port],
            "path": str(mesh_path.resolve()),
            "sha256": hashlib.sha256(mesh_text.encode()).hexdigest(),
        }

    roles = {
        "host-listener": "owns the only boundary listening endpoint, fixed TCP "
                         f"{args.host_port}; accepts container clients",
        "container-client": "client-only TCPv4 (no listening_ports); discovers "
                            f"the host via initialPeersList {address}:{args.host_port}",
    }
    if args.container_mesh:
        roles["container-client-mesh"] = (
            "client-only TCPv4 to the host listener PLUS a netns-local UDPv4 "
            f"transport and SPDP multicast peer {SPDP_MULTICAST_GROUP}:"
            f"{spdp_multicast_port} so container-stack processes discover each "
            "other inside the container; still no TCP listening port"
        )

    report = {
        "schema": REPORT_SCHEMA,
        "transport": "TCPv4",
        "topology": TOPOLOGY,
        "host_address": address,
        "host_port": args.host_port,
        "roles": roles,
        "profiles": entries,
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
