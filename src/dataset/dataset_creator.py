import numpy as np


class DatasetCreator:
    def __init__(self, max_discontiuities, num_samples, num_features, end_time=10):
        self.max_discontiuities = max_discontiuities
        self.num_samples = num_samples
        self.num_features = num_features
        self.end_time = end_time

    def custom_sin(self, x: np.ndarray, a: np.ndarray, b: float, t: np.ndarray) -> np.ndarray:
        """
        Custom sine function with time-dependent amplitude and phase.


        Parameters
        ----------
        x : np.ndarray
            Input values.
        a : np.ndarray
            Amplitude values for each time point.
        b : float
            Phase shift.
        t : np.ndarray
            Time values.

        Returns
        -------
        np.ndarray
            Sine function values.
        """
        return np.sin(np.dot(a, x) + b * t)

    def _create_functions(self, b, threshold):
        """
        Create a dataset with a custom sine function.

        Returns
        -------
        np.ndarray
            Dataset with custom sine function.
        """

        for i in range(1, self.max_discontiuities):
            x = np.random.uniform(-5, 5, self.num_features)
            count = 0
            while count < self.num_samples:
                discontinuities = np.sort(np.random.choice(np.arange(1, self.end_time, 1), i, replace=False))

                a_vector = np.random.rand(i + 1, self.num_features)
                ts = []
                next_start_offeset = self.end_time / 100
                for li in range(len(discontinuities)):
                    if li == 0:
                        ts.append(np.linspace(0, discontinuities[0], discontinuities[0] * 10))
                    else:
                        ts.append(
                            np.linspace(
                                discontinuities[li - 1] + next_start_offeset,
                                discontinuities[li],
                                (discontinuities[li] - discontinuities[li - 1]) * 10,
                            )
                        )

                ts.append(
                    np.linspace(
                        discontinuities[-1] + next_start_offeset,
                        self.end_time,
                        (self.end_time - discontinuities[-1]) * 10,
                    )
                )

                ys = [self.custom_sin(x, a, b, t) for a, t in zip(a_vector, ts)]

                y = np.concatenate(ys)
                t = np.concatenate(ts)

                if all([abs(ys[i + 1][0] - ys[i][-1]) > threshold for i in range(len(ys) - 1)]):
                    count += 1
                    yield x, y, t, discontinuities, a_vector, b
                else:
                    continue

    def create_dataset(self, b=0.5, threshold=0.2):
        """
        Create a dataset with a custom sine function.

        Returns
        -------
        np.ndarray
            Dataset with custom sine function.
        """
        function_dict = {}
        for i, (x, y, t, discontinuities, a_vector, b) in enumerate(self._create_functions(b, threshold)):
            function_dict[i] = {
                "x": x.tolist(),
                "y": y.tolist(),
                "t": t.tolist(),
                "discontinuities": discontinuities.tolist(),
                "a_vector": a_vector.tolist(),
                "b": b,
                "num_discontinuities": len(discontinuities),
            }

        return function_dict
