from google.cloud import storage
from urllib.parse import urlparse


def parse_gs_uri(gs_uri: str) -> tuple[str, str]:
    """
    Split gs://bucket/path into (bucket, blob_path)
    """
    parsed = urlparse(gs_uri)
    if parsed.scheme != "gs":
        raise ValueError(f"Not a valid gs:// URI: {gs_uri}")

    bucket = parsed.netloc
    blob_path = parsed.path.lstrip("/")

    return bucket, blob_path


def copy_gs_uri(source_uri: str, dest_uri: str) -> None:
    """
    Copy a single GCS object using gs:// URIs
    """

    client = storage.Client()

    src_bucket_name, src_blob_name = parse_gs_uri(source_uri)
    dst_bucket_name, dst_blob_name = parse_gs_uri(dest_uri)

    src_bucket = client.bucket(src_bucket_name)
    dst_bucket = client.bucket(dst_bucket_name)

    src_blob = src_bucket.blob(src_blob_name)
    print("read source:")
    src_bucket.copy_blob(src_blob, dst_bucket, dst_blob_name)

    print(f"Copied:\n  {source_uri}\n→ {dest_uri}")


if __name__ == "__main__":
    copy_gs_uri(
        source_uri="gs://ssb-off-fin-data-delt-kostra-befolkning-delt-prod/kommune/folkemengde_kommune_2024.parquet",
        dest_uri="gs://ssb-dapla-felles-data-produkt-prod/kostra/eksempeldata/folkemengde_kommune_2024.parque",
    )