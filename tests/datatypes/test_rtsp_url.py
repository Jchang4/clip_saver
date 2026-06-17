from clip_saver.datatypes.rtsp_url import RtspUrl


def test_rtsp_url_formats_as_rtsp_string():
    rtsp_url = RtspUrl(
        username="admin",
        password="password",
        ip_address="192.168.1.100",
        port=554,
        channel=1,
        subtype=0,
    )

    assert (
        str(rtsp_url)
        == "rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0"
    )


def test_from_rtsp_url_parses_rtsp_string():
    rtsp_url = RtspUrl.from_rtsp_url(
        "rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0"
    )

    assert rtsp_url == RtspUrl(
        username="admin",
        password="password",
        ip_address="192.168.1.100",
        port=554,
        channel=1,
        subtype=0,
    )


def test_from_yolo_path_parses_result_path():
    rtsp_url = RtspUrl.from_yolo_path(
        "rtsp_//admin_password_192.168.1.100_554/cam/realmonitor_channel_1_subtype_0"
    )

    assert rtsp_url == RtspUrl(
        username="admin",
        password="password",
        ip_address="192.168.1.100",
        port=554,
        channel=1,
        subtype=0,
    )
