class BaseTool:
    name = "Base Tool"
    description = "Description of the base tool"

    def __init__(self):
        self.load_error = None
        self.is_callable = True
        try:
            self.verify_setup()
        except Exception as e:
            self.load_error = f"Setup verification failed: {e}"
            self.is_callable = False

    def verify_setup(self):
        """
        Override this method in derived classes to implement setup verification logic.
        """
        pass

    def run(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method")
