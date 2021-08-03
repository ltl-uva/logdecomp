from pathlib import Path
import shutil
import urllib.request


def main():

    path = Path("eigen.zip")
    url = "https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.zip"
    if not path.exists():
        with urllib.request.urlopen(url) as request:
            path.write_bytes(request.read())
    dest = path.stem
    shutil.rmtree(dest, ignore_errors=True)
    shutil.unpack_archive(path, dest)


if __name__ == '__main__':
    main()

