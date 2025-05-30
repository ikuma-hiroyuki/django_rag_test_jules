from django.contrib import admin
from .models import Document

@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'title', 'uploaded_at', 'filename') # Added filename to display
    list_filter = ('user', 'uploaded_at')
    search_fields = ('title', 'file') # Allow searching by title or filename (file.name)
    raw_id_fields = ('user',) # Useful for lots of users

    def filename(self, obj): # To display the result of the model's method
        return obj.filename()
    filename.short_description = 'Filename' # Column header in admin
