"""Keep the checkpoint referenced by latest, regardless of folder naming."""
import elements
class VerifiedCheckpoint(elements.Checkpoint):
    def _cleanup(self):
        if not self._keep:
            return
        latest = (self._directory / 'latest').read_text().strip()
        current = self._directory / latest
        if not (current / 'done').exists():
            raise RuntimeError(f'Refusing cleanup: current checkpoint is incomplete: {current}')
        folders = [p for p in self._directory.glob('*') if p.name != 'latest']
        # Always retain the checkpoint just saved; names of imported checkpoints
        # cannot take precedence over it.
        others = sorted([p for p in folders if p.name != latest])
        for folder in others[:max(0, len(others) - (self._keep - 1))]:
            folder.remove(recursive=True)
    def save(self, path=None, keys=None):
        super().save(path=path, keys=keys)
        current = elements.Path(path) if path else self._directory / (self._directory / 'latest').read_text().strip()
        needed = list(keys) if keys is not None else list(self._saveables)
        for name in ['done'] + [key + '.pkl' for key in needed]:
            if not (current / name).exists():
                raise RuntimeError(f'Checkpoint verification failed: missing {current / name}')
