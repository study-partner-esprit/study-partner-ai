class CoachStateStore:
    """
    Later:
    - Mongo / Redis
    - per user
    """

    def __init__(self):
        self.ignored_count = 0

    def increment_ignored(self):
        self.ignored_count += 1

    def reset_ignored(self):
        self.ignored_count = 0
