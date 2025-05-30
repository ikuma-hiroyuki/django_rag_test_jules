import uuid
from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    # Add any other custom fields if needed in the future
    # For now, inheriting AbstractUser fields is sufficient along with UUID

    def __str__(self):
        return self.username
