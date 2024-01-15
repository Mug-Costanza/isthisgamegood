from django import forms

class GameSearchForm(forms.Form):
    game_title = forms.CharField(label='Enter game title')

