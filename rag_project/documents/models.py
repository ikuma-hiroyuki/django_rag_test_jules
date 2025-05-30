import uuid
import os # For os.path.basename
from django.db import models
from django.conf import settings # To import AUTH_USER_MODEL

# It's good practice to use settings.AUTH_USER_MODEL
# instead of directly importing the User model.

def user_directory_path(instance, filename):
    # file will be uploaded to MEDIA_ROOT/user_<id>/documents/<filename>
    # The user ID for the custom User model is a UUID, ensure it's stringified for path
    return f'user_{str(instance.user.id)}/documents/{filename}'

class Document(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='documents'
    )
    file = models.FileField(upload_to=user_directory_path)
    title = models.CharField(max_length=255, blank=True) # Title can be optional, derived from filename if needed
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title or os.path.basename(self.file.name) # Use basename for __str__ too

    # Optional: Add a method to get the filename
    def filename(self):
        return os.path.basename(self.file.name)
