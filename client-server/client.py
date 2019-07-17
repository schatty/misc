from time import sleep
import requests


def worker():
    host = 'localhost'
    port = 18000

    while True:
        # Make request a
        request = f'http://{host}:{port}/post_a_request/'
        res = requests.post(request).json()
        print("Got from request a: ", res)
        sleep(1)

        # Make request b
        request = f'http://{host}:{port}/post_b_request/'
        res = requests.post(request).json()
        print("Got from request b: ", res)
        sleep(1)


if __name__ == "__main__":
    worker()
