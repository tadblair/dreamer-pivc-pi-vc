"""Online trial rules for the elevated-track task; distances in meters."""
import numpy as np

CORNERS = np.array([[1.18, 1.18], [-1.18, 1.18], [1.18, -1.18], [-1.18, -1.18]])


def start_sequence(rng):
    # Rejection sampling is uniform over balanced sequences satisfying the constraint.
    while True:
        seq = rng.permutation([1] * 16 + [-1] * 16)
        if not any(np.all(seq[i:i+4] == seq[i]) for i in range(29)):
            return seq


class Session:
    def __init__(self, rng, sequence=None):
        self.sequence = start_sequence(rng) if sequence is None else np.asarray(sequence)
        assert len(self.sequence) == 32 and (self.sequence == 1).sum() == 16
        assert set(self.sequence) == {-1, 1}
        assert not any(np.all(self.sequence[i:i+4] == self.sequence[i]) for i in range(29))
        self.config = 0
        self.steps = self.config0_steps = self.completed = 0
        self.correct = self.choices = 0
        self.choice = None
        self.available = True
        self.done = self.terminal = self.fell = False
        self.events = []
        self.trial_choices = []
        self.previous_y = .25 * self.sequence[0]
        self.transitioned = False

    def begin_step(self):
        assert not self.done
        self.steps += 1
        if self.config == 0:
            self.config0_steps += 1
        self.transitioned = False

    def change(self, config):
        old = self.config
        self.config = config
        self.transitioned = True
        self.events.append(dict(step=self.steps, old=old, new=config))
        if config == 0:
            self.config0_steps = 0
            self.available = True
            self.choice = None

    def sample(self, xy, supported=True, fraction=1., body_z=1.0185):
        """Called at every physics substep; at most one door change per control step."""
        x, y = xy
        previous_y = self.previous_y
        self.previous_y = y
        if self.done:
            return 0.
        if not supported:
            self.done = self.terminal = self.fell = True
            return -20.
        # Count first corner choice only during the currently available reward trial.
        if self.available:
            distances = np.linalg.norm(CORNERS - xy, axis=1)
            # Exact sphere-to-solid-cylinder overlap (upright body; meters).
            radial_gap = np.maximum(distances - .02, 0.)
            vertical_gap = max(abs(body_z - 1.005) - .005, 0.)
            contacts = radial_gap**2 + vertical_gap**2 <= .0185**2 + 1e-14
            if self.choice is None and contacts.any():
                self.choice = int(np.flatnonzero(contacts)[0])
                self.choices += 1
                self.correct += int(self.choice == 0)
                self.trial_choices.append(dict(trial=self.completed+1, corner=self.choice, step=self.steps))
            if contacts[0]:
                self.completed += 1
                self.available = False
                if self.completed == 32:
                    self.done = self.terminal = True
                else:
                    self.change(3 if self.sequence[self.completed] == 1 else 4)
                return 1.
        if not self.transitioned:
            if self.config == 0 and self.config0_steps - 1 + fraction >= 40 and abs(y) > .8:
                self.change(2 if self.sequence[self.completed] == 1 else 1)
            elif self.config == 3 and x < .06 and y < .8:
                self.change(0)
            elif self.config == 4 and previous_y >= -.2 and y < -.2 and x != 0:
                self.change(6 if x < 0 else 5)
            elif self.config in (5, 6) and abs(x) < .06 and y > -.8:
                self.change(0)
        return 0.

    def end_step(self):
        if self.steps >= 7680:
            self.done = True  # Time limit is a truncation, not a terminal fall/success.

    @property
    def accuracy(self):
        return 100. * self.correct / self.choices if self.choices else 0.

