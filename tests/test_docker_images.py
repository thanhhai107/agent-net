from nika.net_env.utils.docker_files import docker_images


def test_p4_int_influxdb_image_has_local_dockerfile() -> None:
    dockerfile = docker_images._dockerfile_for_image("kathara/influxdb")

    assert dockerfile.name == "Dockerfile"
    assert dockerfile.parent.name == "p4_int"
    assert dockerfile.is_file()


def test_ensure_local_images_builds_missing_influxdb(monkeypatch) -> None:
    built: list[str] = []

    def fake_image_exists(image: str) -> bool:
        return image != "kathara/influxdb" or image in built

    monkeypatch.setattr(docker_images, "image_exists", fake_image_exists)
    monkeypatch.setattr(docker_images, "build_nika_image", built.append)

    docker_images.ensure_nika_docker_images(
        [
            "kathara/base",
            "kathara/p4",
            "kathara/influxdb",
        ]
    )

    assert built == ["kathara/influxdb"]
