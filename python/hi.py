master_password = input("What is your master password? ")

def view():
    pass

def add():
    name = input("Account Name: ")
    password = input("Password: ")

    with open("passwords.txt", "a") as f: 
        f.write(name + "|" + password)

while True:
    mode = input("Would you like to add a new password or view existing ones (view, add)? ").lower()
    if mode == "q":
        break

    if mode == "view":
        view()

    elif mode == "add":
        add()

    else:
        print("Invalid mode.")
        continue
