import argparse
import base64
import mimetypes
import os

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_VISION_MODEL = "openrouter/free"


def _get_client() -> OpenAI:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )


def describe_image_and_rewrite_query(
    image_path: str,
    query: str,
    model: str = DEFAULT_VISION_MODEL,
) -> tuple[str, int | None]:
    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"

    with open(image_path, "rb") as f:
        img = f.read()

    data_url = f"data:{mime};base64,{base64.b64encode(img).decode()}"

    system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""

    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt.strip()},
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": query.strip()},
            ],
        }
    ]

    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    content = response.choices[0].message.content if response.choices else None
    if content is None:
        raise RuntimeError("No text in API response")

    total_tokens = (
        response.usage.total_tokens if response.usage is not None else None
    )
    return content.strip(), total_tokens


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite a movie search query using an image and a multimodal LLM"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image file",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Text query to rewrite based on the image",
    )

    args = parser.parse_args()
    rewritten_query, total_tokens = describe_image_and_rewrite_query(
        args.image,
        args.query,
    )

    print(f"Rewritten query: {rewritten_query}")
    if total_tokens is not None:
        print(f"Total tokens:    {total_tokens}")


if __name__ == "__main__":
    main()