from typing import List, TypeVar

T = TypeVar("T")


def cap_text(text: str, max_chars: int) -> str:
    """Truncate text to max_chars. A max_chars of 0 (or less) means no limit."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars]


def cap_list(items: List[T], max_items: int) -> List[T]:
    """Take the first max_items. A max_items of 0 (or less) means all items."""
    if max_items <= 0:
        return items
    return items[:max_items]


if __name__ == "__main__":
    assert cap_text("hello world", 5) == "hello"
    assert cap_text("hello", 0) == "hello"          # 0 = no limit
    assert cap_text("hi", 10) == "hi"
    assert cap_list([1, 2, 3, 4], 2) == [1, 2]
    assert cap_list([1, 2, 3], 0) == [1, 2, 3]      # 0 = all
    print("OK")
