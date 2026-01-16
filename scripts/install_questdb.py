#!/usr/bin/env python3
"""
Install QuestDB time-series database.
Run with: sudo python3 install_questdb.py
"""

import os
import subprocess
import sys
import tarfile
import urllib.request
import shutil
from pathlib import Path

QUESTDB_VERSION = "8.2.1"
QUESTDB_URL = f"https://github.com/questdb/questdb/releases/download/{QUESTDB_VERSION}/questdb-{QUESTDB_VERSION}-no-jre-bin.tar.gz"
INSTALL_DIR = Path("/opt/questdb")
DATA_DIR = Path("/var/lib/questdb")
QUESTDB_USER = "questdb"


def check_root():
    if os.geteuid() != 0:
        print("Error: This script must be run as root (use sudo)")
        sys.exit(1)


def check_java():
    """Check if Java 11+ is installed."""
    try:
        result = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            text=True
        )
        # Java version info goes to stderr
        version_output = result.stderr
        print(f"Java detected: {version_output.splitlines()[0]}")
        return True
    except FileNotFoundError:
        print("Warning: Java not found. QuestDB requires Java 11+")
        print("Install with: sudo apt install openjdk-17-jre-headless")
        return False


def download_questdb(dest_path: Path):
    """Download QuestDB tarball."""
    print(f"Downloading QuestDB {QUESTDB_VERSION}...")
    urllib.request.urlretrieve(QUESTDB_URL, dest_path)
    print(f"Downloaded to {dest_path}")


def extract_questdb(tarball: Path, dest: Path):
    """Extract QuestDB to installation directory."""
    print(f"Extracting to {dest}...")

    if dest.exists():
        print(f"Removing existing installation at {dest}")
        shutil.rmtree(dest)

    dest.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tarball, "r:gz") as tar:
        # Extract to temp location first
        tar.extractall(dest.parent)

    # Move from extracted dir to final location
    extracted = dest.parent / f"questdb-{QUESTDB_VERSION}-no-jre-bin"
    if extracted.exists():
        for item in extracted.iterdir():
            shutil.move(str(item), str(dest))
        extracted.rmdir()


def create_user():
    """Create questdb system user if it doesn't exist."""
    try:
        subprocess.run(["id", QUESTDB_USER], capture_output=True, check=True)
        print(f"User {QUESTDB_USER} already exists")
    except subprocess.CalledProcessError:
        print(f"Creating system user {QUESTDB_USER}...")
        subprocess.run([
            "useradd", "-r", "-s", "/bin/false", "-d", str(DATA_DIR), QUESTDB_USER
        ], check=True)


def setup_directories():
    """Create data directory and set permissions."""
    print(f"Setting up data directory at {DATA_DIR}...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Create conf directory
    conf_dir = DATA_DIR / "conf"
    conf_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(["chown", "-R", f"{QUESTDB_USER}:{QUESTDB_USER}", str(DATA_DIR)], check=True)
    subprocess.run(["chown", "-R", f"{QUESTDB_USER}:{QUESTDB_USER}", str(INSTALL_DIR)], check=True)


def create_server_config():
    """Create server.conf with network isolation (localhost only)."""
    config_content = """# QuestDB Server Configuration
# Network Isolation: All interfaces bound to localhost only

# HTTP/REST API - Web Console
http.bind.to=127.0.0.1:9000
http.min.net.bind.to=127.0.0.1:9003

# PostgreSQL Wire Protocol
pg.net.bind.to=127.0.0.1:8812

# InfluxDB Line Protocol (ILP) over TCP
line.tcp.net.bind.to=127.0.0.1:9009

# InfluxDB Line Protocol (ILP) over UDP
line.udp.bind.to=127.0.0.1:9009
line.udp.receive.buffer.size=4194304

# Health check endpoint
http.health.check.authentication.required=false
"""

    conf_path = DATA_DIR / "conf" / "server.conf"
    print(f"Creating server config with network isolation at {conf_path}...")
    conf_path.write_text(config_content)

    subprocess.run(["chown", f"{QUESTDB_USER}:{QUESTDB_USER}", str(conf_path)], check=True)


def create_systemd_service():
    """Create systemd service file."""
    service_content = f"""[Unit]
Description=QuestDB Time-Series Database
After=network.target

[Service]
Type=simple
User={QUESTDB_USER}
Group={QUESTDB_USER}
ExecStart={INSTALL_DIR}/bin/questdb.sh start -d {DATA_DIR} -n
ExecStop={INSTALL_DIR}/bin/questdb.sh stop
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
"""

    service_path = Path("/etc/systemd/system/questdb.service")
    print(f"Creating systemd service at {service_path}...")
    service_path.write_text(service_content)

    subprocess.run(["systemctl", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "enable", "questdb"], check=True)


def create_symlink():
    """Create symlink for questdb command."""
    symlink = Path("/usr/local/bin/questdb")
    target = INSTALL_DIR / "bin" / "questdb.sh"

    if symlink.exists():
        symlink.unlink()

    symlink.symlink_to(target)
    print(f"Created symlink: {symlink} -> {target}")


def main():
    check_root()

    if not check_java():
        response = input("Continue without Java? QuestDB will not run. [y/N]: ")
        if response.lower() != 'y':
            sys.exit(1)

    tarball = Path("/tmp/questdb.tar.gz")

    try:
        download_questdb(tarball)
        extract_questdb(tarball, INSTALL_DIR)
        create_user()
        setup_directories()
        create_server_config()
        create_systemd_service()
        create_symlink()

        print("\n" + "=" * 50)
        print("QuestDB installation complete!")
        print("=" * 50)
        print(f"\nInstalled to: {INSTALL_DIR}")
        print(f"Data directory: {DATA_DIR}")
        print(f"Config file: {DATA_DIR}/conf/server.conf")
        print("\nSECURITY: Network isolation enabled (localhost only)")
        print("  - All interfaces bound to 127.0.0.1")
        print("  - Remote access requires SSH tunnel or reverse proxy")
        print("\nCommands:")
        print("  sudo systemctl start questdb   # Start service")
        print("  sudo systemctl stop questdb    # Stop service")
        print("  sudo systemctl status questdb  # Check status")
        print("\nEndpoints (localhost only):")
        print("  Web Console: http://127.0.0.1:9000")
        print("  PostgreSQL Wire: 127.0.0.1:8812")
        print("  InfluxDB Line Protocol: 127.0.0.1:9009")
        print("\nRemote access via SSH tunnel:")
        print("  ssh -L 9000:127.0.0.1:9000 user@server")

    finally:
        if tarball.exists():
            tarball.unlink()


if __name__ == "__main__":
    main()
