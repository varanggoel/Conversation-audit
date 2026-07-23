import re

WHATSAPP_MESSAGE_PATTERN = re.compile(
    r'^(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s*(?P<time>\d{1,2}:\d{2}(?:\s?[APMapm]{2})?)\s*-\s*(?P<sender>[^:]+):\s*(?P<text>.*)$'
)


def parse_chat(chat_text):
    messages = []
    current = None

    for line in chat_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        match = WHATSAPP_MESSAGE_PATTERN.match(stripped)
        if match:
            if current:
                messages.append(current)

            current = {
                "id": len(messages) + 1,
                "date": match.group("date"),
                "time": match.group("time"),
                "sender": match.group("sender"),
                "text": match.group("text"),
            }
        elif current:
            current["text"] += "\n" + stripped

    if current:
        messages.append(current)

    return messages