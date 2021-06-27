from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles



templates = Jinja2Templates(directory='templates')

async def homepage(request):
    print(request.query_params)
    return templates.TemplateResponse('index.html', {'request': request})

routes = [
    Route('/', endpoint=homepage),
    Mount('/static', app=StaticFiles(directory='static'), name="static"),
]

app = Starlette(debug=True, routes=routes)
