from django import forms
from .models import MediaFile

class UploadFileForm(forms.ModelForm):
    class Meta:
        model = MediaFile
        fields = ['file']