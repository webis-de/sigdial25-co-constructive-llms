# Co-Constructive LLMs study interface

## Main technical setup
- Django as main webserver framework
- SQLite as databse
- KISSKI inference cluster as LLM backend
- TailwindCSS as frontend CSS framework


## Development setup
1. Copy `study_app/study_app/secrets.example.py` to `study_app/study_app/secrets.py` and fill the values accordingly.
2. Setup new python environment and install the neccesary packages using `pipenv install`.
3. Initialize migrations using `python manage.py makemigrations`.
4. Initialize a database using `python manage.py migrate`.
5. Start django server with `python manage.py runserver`.

