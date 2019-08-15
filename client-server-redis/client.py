from time import sleep
import redis


if __name__ == "__main__":
    host = '104.197.23.62'
    port = 19000
    db = 0
    r = redis.Redis(host=host, port=port, db=db)

    print("Starting Client...")
    while True:
        n = r.get("my_number")
        print("My number: ", n)
        sleep(1)
