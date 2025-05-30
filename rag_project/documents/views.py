from django.shortcuts import render, redirect
from django.views.generic import CreateView, ListView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib import messages # For user feedback
from .models import Document
from .forms import DocumentForm
from . import rag_service # Import the RAG service
from django.urls import reverse_lazy
import os

class DocumentUploadView(LoginRequiredMixin, CreateView):
    model = Document
    form_class = DocumentForm
    template_name = 'documents/document_form.html' # We'll create this template later
    success_url = reverse_lazy('documents:document_list') # Redirect to list view after successful upload

    def form_valid(self, form):
        form.instance.user = self.request.user
        if not form.instance.title:
            # Use os.path.basename to get only the filename, not the full path
            form.instance.title = os.path.basename(form.instance.file.name)
        
        # Let the CreateView handle saving the object by calling super().form_valid() first.
        # This sets self.object to the saved instance.
        response = super().form_valid(form) 
        
        # Now, if the object was successfully created, process it for RAG.
        if self.object:
            try:
                print(f"DocumentUploadView: Attempting to add document {self.object.id} to RAG store for user {self.request.user.id}")
                # Ensure user.id is a string, especially if it's a UUID.
                rag_service.add_document_to_user_store(
                    user_id=str(self.request.user.id), 
                    document_model_instance=self.object
                )
                messages.success(self.request, f"Document '{self.object.title}' uploaded and sent for RAG processing.")
            except Exception as e:
                error_message = f"Document '{self.object.title}' uploaded, but RAG processing failed: {e}"
                print(f"DocumentUploadView: {error_message}")
                messages.error(self.request, error_message)
                # Optional: Decide if self.object should be deleted if RAG processing is critical.
                # For now, we'll let it exist in the database and show an error.
                
        return response

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['view_title'] = 'Upload Document'
        return context

class DocumentListView(LoginRequiredMixin, ListView):
    model = Document
    template_name = 'documents/document_list.html' # We'll create this template later
    context_object_name = 'documents'

    def get_queryset(self):
        # Only list documents for the currently logged-in user
        return Document.objects.filter(user=self.request.user).order_by('-uploaded_at')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['view_title'] = 'My Documents'
        return context
