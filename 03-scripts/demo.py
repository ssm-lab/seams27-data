import zmq

print("Starting demo...")

context = zmq.Context()
socket = context.socket(zmq.REQ)

print("Connecting...")
socket.connect("tcp://127.0.0.1:5558")
print("Connected.")

print("Sending initialize...")
socket.send_json({
    "type": "initialize"
})
print("Initialize sent. Waiting for response...")

response = socket.recv_json()

print("Initialize response:")
print(response)

# 2. Send episode outcomes
for episode in range(1, 201):
    socket.send_json({
        "type": "episodeOutcome",
        "episode": episode,
        "episodicReturn": -1.0,
        "success": False
    })

    response = socket.recv_json()
    print(f"Episode {episode}: {response}")

    instruction = response["instruction"]

    if instruction == "PROCEED_TO_ENVIRONMENT":
        print("Adaptation triggered!")
        print("New environment:")
        print(response["environment"])

    elif instruction == "STOP":
        print("Training stopped.")
        break

socket.close()
context.term()