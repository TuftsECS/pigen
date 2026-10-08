class DevicePhysics:
    """Physics core of one memory device.

    A device package (e.g. ``rram/``) subclasses this and exposes the subclass as ``Physics`` in its
    ``__init__.py``; ``train.py --physics <package>`` then trains the shared models in ``core``
    against it.
    """

    # Name of the internal state variable driven by the physics, used in plots and logs.
    state_name = 'state'
    # Material names in dataset order; a material's position is its embedding index.
    materials = ()
    # Per material, voltages that a stride split always keeps in the training set (e.g. switching thresholds).
    split_voltages = {}
    # Time [s] before the first sample of a sequence, used to compute the first time step.
    initial_time = 0.0

    def simulate(self, dt, voltage, current, material):
        """Physical internal state at every time step of one sequence.

        All inputs are 1-D tensors in physical units with the same length; ``material`` is a name from
        ``materials``. Returns a tensor of the same length.
        """
        raise NotImplementedError

    def normalize_state(self, state):
        """Map the physical state onto [-1, 1], the output range of the state PINN."""
        raise NotImplementedError
