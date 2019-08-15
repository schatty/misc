from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.response import Response


def post_a_request(request):
    print("Post a request")
    json = {'n': 1}
    return Response(json=json)


def post_b_request(request):
    print("Post b request")
    json = {'n': 2}
    return Response(json=json)



if __name__ == "__main__":
    host = '0.0.0.0'
    port = 18000

    with Configurator() as config:
        config.add_route('post_a_request', '/post_a_request/')
        config.add_view(post_a_request, route_name='post_a_request', request_method='POST')

        config.add_route('post_b_request', '/post_b_request/')
        config.add_view(post_b_request, route_name='post_b_request', request_method='POST')

        app = config.make_wsgi_app()


    print("Starting server...")
    server = make_server(host, port, app)
    server.serve_forever()
