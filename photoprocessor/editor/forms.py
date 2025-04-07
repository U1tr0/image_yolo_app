from django import forms
from .models import MediaFile

class UploadFileForm(forms.ModelForm):
    class Meta:
        model = MediaFile
        fields = ['file']

class FilterForm(forms.Form):
    FILTER_CHOICES = [
        ('grayscale', 'Чёрно-белый'),
        ('blur', 'Размытие'),
    ]
    filter_type = forms.ChoiceField(choices=FILTER_CHOICES)