from time import sleep
import redis


if __name__ == "__main__":
    host = '0.0.0.0'
    port = 19000
    db = 0
    r = redis.Redis(host=host, port=port, db=db)

    print("Starting server...")
    counter = 0
    while True:
        r.set("my_number", counter)
        counter += 1
        sleep(1)
