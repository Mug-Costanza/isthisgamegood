from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views import View
from .forms import GameSearchForm
import subprocess
import json

class Index(View):
    template_name = 'index.html'

    def get(self, request, *args, **kwargs):
        form = GameSearchForm()
        return render(request, self.template_name, {'form': form})

    def post(self, request, *args, **kwargs):
        form = GameSearchForm(request.POST)
        if form.is_valid():
            game_title = form.cleaned_data['game_title']

            num_reviews = 5  # You can change this to the desired number of reviews

            # Call the Python script
            script_path = 'steam.py'  # Replace with the actual path
            command = ['python3', script_path, game_title, str(num_reviews)]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            print("Return Code:", process.returncode)

            if process.returncode == 0:
                # Check if stdout is not empty before decoding
                print("This works!")
                reviews = json.loads(stdout.decode('utf-8'))
                return render(request, self.template_name, {'form': form, 'game_info': {'title': game_title, 'reviews': reviews}})
            else:
                error_message = f"Error: {stderr.decode('utf-8')}"

            return render(request, self.template_name, {'form': form, 'error_message': error_message})

        return render(request, self.template_name, {'form': form})

class About(View):
    def get(self, request, *args, **kwargs):
        return render(request, 'about.html')

class Privacy(View):
    def get(self, request, *args, **kwargs):
        return render(request, 'privacy.html')

class Terms(View):
    def get(self, request, *args, **kwargs):
        return render(request, 'terms.html')


