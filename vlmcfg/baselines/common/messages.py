"""Chat-message helpers shared by the baselines."""


def drop_visuals(messages: list[dict]) -> list[dict]:
    """The same chat messages without image or video entries (the image-free prompt).

    Filters the message structure and lets the chat template render it, rather than
    deleting image markers from rendered text.
    """
    out = []
    for message in messages:
        content = [
            dict(c)
            for c in message["content"]
            if c.get("type") not in ("image", "video")
        ]
        if not content:
            raise ValueError(
                "the image-free prompt would be empty: the message has no text"
            )
        out.append({**message, "content": content})
    return out
