"""Publisher byte identities and exclusive acquisition, without network/media/model calls."""
import hashlib
import io

import pytest

from scripts import acquire_video_common_sources_v91 as acquisition


class Budget:
    def check(self):
        pass


def test_git_lfs_pointer_supplies_exact_digest_and_size():
    digest = 'a' * 64
    data = f'version https://git-lfs.github.com/spec/v1\noid sha256:{digest}\nsize 123\n'.encode()
    assert acquisition.lfs_pointer(data) == (digest, 123)
    with pytest.raises(ValueError, match='Expected immutable'):
        acquisition.lfs_pointer(b'<html>not a pointer</html>')
    with pytest.raises(ValueError, match='Complete LFS'):
        acquisition.lfs_pointer(b'version https://git-lfs.github.com/spec/v1\nsize 123\n')


@pytest.mark.parametrize('size,digest,lfs_size', [(True, 'a'*64, True), (100, None, 100), (100, 'a'*64, 99),
    (acquisition.MAX_ASSET+1, 'a'*64, acquisition.MAX_ASSET+1)])
def test_inconsistent_or_unbounded_publisher_descriptors_are_rejected(size, digest, lfs_size):
    with pytest.raises(ValueError, match='Publisher LFS'):
        acquisition.hf_object({'size': size, 'lfs': {'sha256': digest, 'size': lfs_size}})


def asset(data, *, digest=None):
    return {'index': 0, 'file': 'asset-000.mp4', 'url': 'https://example.invalid/source',
            'bytes': len(data), 'sha256': digest or hashlib.sha256(data).hexdigest()}


def test_exact_download_and_existing_file_are_never_silently_reused(tmp_path, monkeypatch):
    data = b'authored byte fixture, not a decoded video'
    calls = []
    def opened(request, timeout):
        calls.append(request.full_url)
        return io.BytesIO(data)
    monkeypatch.setattr(acquisition, 'urlopen', opened)
    path = tmp_path/'asset-000.mp4'
    result = acquisition.download(asset(data), path, Budget())
    assert result['status'] == 'byte_verified' and path.read_bytes() == data
    with pytest.raises(FileExistsError):
        acquisition.download(asset(data), path, Budget())
    assert len(calls) == 1 and path.read_bytes() == data


def test_digest_mismatch_and_oversize_failure_are_retained(tmp_path, monkeypatch):
    data = b'original bytes'
    monkeypatch.setattr(acquisition, 'urlopen', lambda *a, **k: io.BytesIO(data))
    path = tmp_path/'asset-000.mp4'
    with pytest.raises(ValueError, match='digest/size mismatch'):
        acquisition.download(asset(data, digest='b'*64), path, Budget())
    assert path.read_bytes() == data
    path.unlink()  # isolated authored fixture only; production never deletes partial acquisitions
    wrong = asset(data)
    wrong['bytes'] = 1
    with pytest.raises(ValueError, match='exceeds publisher'):
        acquisition.download(wrong, path, Budget())
    assert path.exists()
