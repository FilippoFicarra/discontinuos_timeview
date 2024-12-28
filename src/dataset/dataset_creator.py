import numpy as np


class DatasetCreator:
    def __init__(self, max_discontiuities, num_samples, num_features, end_time=10, linspace=100):
        self.max_discontiuities = max_discontiuities
        self.num_samples = num_samples
        self.num_features = num_features
        self.end_time = end_time
        self.linspace = linspace

        self.custom_functions = {
            "sin": self.custom_sin,
            "linear": self.custom_linear,
            "square": self.custom_square,
        }

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

    def custom_linear(self, x: np.ndarray, a: np.ndarray, b: float, t: np.ndarray) -> np.ndarray:
        """
        Custom linear function with time-dependent slope and intercept.


        Parameters
        ----------
        x : np.ndarray
            Input values.
        a : np.ndarray
            Slope values for each time point.
        b : float
            Intercept.
        t : np.ndarray
            Time values.

        Returns
        -------
        np.ndarray
            Linear function values.
        """
        return np.dot(a, x) + b * t

    def custom_square(self, x: np.ndarray, a: np.ndarray, b: float, t: np.ndarray) -> np.ndarray:
        """
        Custom square function with time-dependent amplitude and phase.


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
            Square function values.
        """
        
        
        return np.dot(a, x) + b * t

    def _create_functions(self, b, threshold):
        """
        Create a dataset with a custom sine function.

        Returns
        -------
        np.ndarray
            Dataset with custom sine function.
        """
        for function_name, funct in self.custom_functions.items():
            for i in range(1, self.max_discontiuities):
                x = np.random.uniform(-5, 5, self.num_features)
                count = 0
                while count < self.num_samples:
                    discontinuities = np.sort(np.random.choice(np.arange(1, self.end_time, 1), i, replace=False))

                    a_vector = np.random.rand(i + 1, self.num_features)
                    ts = []
                    next_start_offeset = self.end_time / self.linspace

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

                    ys = [funct(x, a, b, t) for a, t in zip(a_vector, ts)]

                    y = np.concatenate(ys)
                    t = np.concatenate(ts)

                    if all([abs(ys[x + 1][0] - ys[x][-1]) > threshold for x in range(len(ys) - 1)]):
                        count += 1
                        yield count-1, function_name, x, y, t, discontinuities, a_vector, b
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
        for count, function_name, x, y, t, discontinuities, a_vector, b in self._create_functions(b, threshold):
            if function_name not in function_dict:
                function_dict[function_name] = {}  
            if f"{len(discontinuities)}_discontinuities" not in function_dict[function_name]:
                function_dict[function_name][f"{len(discontinuities)}_discontinuities"] = {}
                
            function_dict[function_name][f"{len(discontinuities)}_discontinuities"][count] = {
                    "x": x.tolist(),
                    "y": y.tolist(),
                    "t": t.tolist(),
                    "discontinuities": discontinuities.tolist(),
                    "a_vector": a_vector.tolist(),
                    "b": b,
                    "num_discontinuities": len(discontinuities),
                }


        return function_dict
