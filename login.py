

from tkinter import *
from tkinter import messagebox
import mysql.connector
import heapq
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.figure import Figure
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import messagebox
from PIL import Image, ImageTk
from tkinter import messagebox
import heapq
import random

root = Tk()
root.title('Login')
root.geometry('925x500+700+200')
root.configure(bg="white")
root.resizable(False, False)

db_url = "localhost"
db_username = "root"
db_password = "gugan"
database = "hi"

graph = {
    'A': {'B': 1, 'C': 4, 'E': 5},  
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1},
    'E': {'A': 5, 'C': 1}
}


try:
    connection = mysql.connector.connect(
        host=db_url, user=db_username, password=db_password, database=database
    )
    if connection.is_connected():
        cursor = connection.cursor(buffered=True)
        print("Connected to the database")
    else:
        raise mysql.connector.Error("Failed to connect to the database")
except mysql.connector.Error as err:
    print(f"Error: {err}")
    messagebox.showerror("Database Connection Error", f"Error connecting to the database: {err}")
    root.destroy()

logged_in_user = None
global p2
def logout():
    global logged_in_user
    logged_in_user = None
    for widget in root.winfo_children():
        widget.destroy()

    create_login_screen()

def pro():
    global logged_in_user

    for widget in root.winfo_children():
        widget.destroy()

    p1 = Frame(root, width=2000, height=500, bg='light blue')
    p1.place(x=0, y=0)

    heading = Label(p1, text='Profile', fg='black', bg='light blue', font=('Times New Roman', 23, 'bold'))
    heading.place(x=300, y=20)

    if logged_in_user:
        user_info = f"Username: {logged_in_user[1]}\n\nPassword: {logged_in_user[2]}\n\nGender: {logged_in_user[3]}"
        user_info_label = Label(p1, text=user_info, fg='black', bg='light blue', font=('Times New Roman', 14))
        user_info_label.place(x=200, y=80)
    
    Button(p1, width=20, pady=7, text='Log out', bg='brown', font=('Times New Roman', 10, 'bold'), fg='white', border=3, command=logout).place(x=220, y=250)
    Button(p1, width=20, pady=7, text='Home', bg='brown', font=('Times New Roman', 10, 'bold'), fg='white', border=3, command=home).place(x=220, y=300)
    img = PhotoImage(file='prof.png',width=1000,height=700)
    Label(p1,width=1000,height=700,image=img, bg='light blue').place(x=500,y=150)
    root.mainloop()
#dijkstra algorithm-----------------------


def dijkstra(graph, start):
    priority_queue = [(0, start)]
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    previous_nodes = {node: None for node in graph}
    
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        
        if current_distance > distances[current_node]:
            continue
        
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))
            elif distance == distances[neighbor] and random.choice([True, False]):
                
                previous_nodes[neighbor] = current_node
    
    return distances, previous_nodes

def reconstruct_path(previous_nodes, start, end):
    path = []
    current_node = end
    
    while current_node != start:
        if current_node is None:
            return None  
        path.append(current_node)
        current_node = previous_nodes[current_node]
    
    path.append(start)
    path.reverse()
    return path

def create_window(graph, parent):
    for widget in parent.winfo_children():
        widget.destroy()

    home_button = tk.Button(parent, text="Home", command=home, bg='lightblue', fg='black', font=('Times New Roman', 10, 'bold'), border=2)
    home_button.grid(row=0, column=0, padx=5, pady=5)

    start_node_label = tk.Label(parent, text="Start Node:")
    start_node_label.grid(row=1, column=0, padx=5, pady=5)
    start_entry = tk.Entry(parent)
    start_entry.grid(row=1, column=1, padx=5, pady=5)

    end_node_label = tk.Label(parent, text="End Node:")
    end_node_label.grid(row=2, column=0, padx=5, pady=5)
    end_entry = tk.Entry(parent)
    end_entry.grid(row=2, column=1, padx=5, pady=5)

    def on_submit():
        start_node = start_entry.get().upper()
        end_node = end_entry.get().upper()

        if start_node not in graph or end_node not in graph:
            result_label.config(text="Invalid start or end node.")
        else:
            distances, previous_nodes = dijkstra(graph, start_node)
            path = reconstruct_path(previous_nodes, start_node, end_node)
            if path:
                shortest_path = [(path[i], path[i+1]) for i in range(len(path)-1)]
                result_label.config(text=f"Shortest path from {start_node} to {end_node}: {', '.join(path)}\nDistance: {distances[end_node]}")
                plot_graph_tk(graph, shortest_path, start_node, end_node, parent)
                messagebox.showinfo("Shortest Path", f"The shortest path from {start_node} to {end_node} is: {', '.join(path)}\nDistance: {distances[end_node]}")
            else:
                result_label.config(text=f"No path exists between {start_node} and {end_node}.")

    submit_button = tk.Button(parent, text="Submit", command=on_submit)
    submit_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

    result_label = tk.Label(parent, text="")
    result_label.grid(row=4, column=0, columnspan=2, padx=5, pady=5)

    plot_graph_tk(graph, [], "", "", parent)

    parent.mainloop()

def plot_graph_tk(graph, shortest_path, start_node, end_node, window):
    G = nx.Graph()
    
    for node, neighbors in graph.items():
        G.add_node(node)
        for neighbor, weight in neighbors.items():
            G.add_edge(node, neighbor, weight=weight)
    
    pos = nx.spring_layout(G, seed=42)
    
    fig, ax = plt.subplots()
    fig, ax = plt.subplots(figsize=(6, 4)) 
    
    nx.draw(G, pos, ax=ax, with_labels=True, node_size=700, node_color='lightblue', font_size=12, font_weight='bold')
    
    
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    
    nx.draw_networkx_edges(G, pos, edgelist=shortest_path, edge_color='red', width=2)
    
    rect = plt.Rectangle((-1, -1), 2, 9, color='green', zorder=-10, transform=ax.transAxes)
    ax.add_patch(rect)
    
    for widget in window.winfo_children():
        if isinstance(widget, FigureCanvasTkAgg):
            widget.get_tk_widget().grid_forget()
    print("Shortest path:", shortest_path)

    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().grid(row=4, column=0 ,columnspan=11, padx=5, pady=5, sticky="nsew")  


def dist():
    for widget in root.winfo_children():
        widget.destroy()

    p2 = Frame(root, width=100, height=500, bg='pink')
    p2.place(x=0, y=0)

    
    create_window(graph, p2)
    create_window(graph, p2)
#end djk--------------------------------------------------------------
#divide and con-------------------------------------------------------
class HashTable:
    def __init__(self, size):
        self.size = size
        self.table = [[] for _ in range(size)]
        self.conn = self.connect_to_db()
        self.create_table()
        

    def connect_to_db(self):
        return mysql.connector.connect(
            host="localhost",
            user="root",
            password="gugan",
            database="hi"
        )

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        party VARCHAR(255),
        hash_code BIGINT,
        user VARCHAR(255),
        voter_id VARCHAR(255)
        )
        """)
        self.conn.commit()

    def insert(self, key, value, user_index, user_name):
        hash_code = self.hash_function(value)
        if not self.check_duplicate(value):
            self.table[user_index].append((hash_code, value))
            self.store_in_db(key, hash_code, user_name, value)
            
            print(f"Voter ID {value} connected to {self.get_party_name(key)} with hash code: {hash_code}")
            if key == 1:
                messagebox.showinfo("Values", "\n".join(["You voted for DMK"]))
            if key == 2:
                messagebox.showinfo("Values", "\n".join(["You voted for BJP"]))
            if key == 3:
                messagebox.showinfo("Values", "\n".join(["You voted for ADMK"]))

        else:
            messagebox.showinfo("Values", "\n".join(["Voted already"]))

    def store_in_db(self, key, hash_code, user_name, voter_id):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO users (party, hash_code, username, voter_id) 
            VALUES (%s, %s, %s, %s)
            """, (self.get_party_name(key), hash_code, user_name, voter_id))
        self.conn.commit()


    def check_duplicate(self, value):
        
        for bucket in self.table:
            for _, voter_id in bucket:
                if voter_id == value:
                    return True

        
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM Voters WHERE voter_id = %s", (value,))
        result = cursor.fetchone()
        return result[0] > 0

    def display(self):
        for i in range(self.size):
            print(f"Bucket {i} ({self.get_party_name(i+1)}): {self.table[i]}")

    def sort_table(self):
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT party, voter_id FROM users")
        data = cursor.fetchall()
    
        sorted_data = self.merge_sort(data)
    
        for bucket in self.table:
            bucket.clear()
    
        for party, voter_id in sorted_data:
            key = self.get_party_key(party)
            hash_code = self.hash_function(voter_id)
            user_index = key - 1
            self.table[user_index].append((hash_code, voter_id))

    def merge_sort(self, data):
        if len(data) <= 1:
            return data
    
        mid = len(data) // 2
        left_half = data[:mid]
        right_half = data[mid:]
    
        left_half = self.merge_sort(left_half)
        right_half = self.merge_sort(right_half)
    
        return self.merge(left_half, right_half)

    def merge(self, left, right):
        merged = []
        left_index = right_index = 0
    
        while left_index < len(left) and right_index < len(right):
            if self.get_party_key(left[left_index][0]) <= self.get_party_key(right[right_index][0]):
                merged.append(left[left_index])
                left_index += 1
            else:
                merged.append(right[right_index])
                right_index += 1
    
        merged.extend (left[left_index:])
        merged.extend(right[right_index:])
    
        return merged


    @staticmethod
    def get_party_key(party_name):
        parties = {
            "DMK": 1,
            "ADMK": 2,
            "BJP": 3
        }
        return parties.get(party_name, -1)
           
    def fetch_data_from_db(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT party, voter_id FROM users")
        data = cursor.fetchall()
        return data
    def fetch_vote_counts(self):
        cursor = self.conn.cursor()
        cursor.execute("""
        SELECT party, COUNT(*) as vote_count 
        FROM users 
        GROUP BY party
        """)
        data = cursor.fetchall()
        vote_counts = {party: count for party, count in data}
        return vote_counts


    def plot_graph(self):
        data = self.fetch_data_from_db()
        graph = {}
        for party, voter_id in data:
            if party not in graph:
                graph[party] = []
            graph[party].append(voter_id)

        G = nx.Graph()

        central_node = "NGP"
        G.add_node(central_node)

        for party, voters in graph.items():
            G.add_edge(central_node, party)
            for voter in voters:
                G.add_edge(party, voter)

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000, font_size=10, font_weight='bold')
        plt.show()

    @staticmethod
    def get_party_name(key):
        parties = {
            1: "DMK",
            2: "ADMK",
            3: "BJP"
        }
        return parties.get(key, "Unknown")

    @staticmethod
    def hash_function(value):
        return hash(value)

class App:
    def __init__(self, root):
        self.root = root
        
        self.size = 3
        self.hash_table = HashTable(self.size)
        
        self.create_widgets()

    def create_widgets(self):
        frame = ttk.Frame(self.root, padding="200")
        frame.grid(row=0, column=0)

        ttk.Label(frame, text="Enter Your Preference \n1.DMK,\n2.ADMK,\n3.BJP:").grid(row=0, column=0, sticky="w")
        self.preference_entry = ttk.Entry(frame)
        self.preference_entry.grid(row=0, column=1)
        

        ttk.Label(frame, text="Enter your voter ID no:").grid(row=1, column=0, sticky="w")
        self.voter_id_entry = ttk.Entry(frame)
        self.voter_id_entry.grid(row=1, column=1)

        self.submit_button = ttk.Button(frame, text="Home", command=home)
        self.submit_button.grid(row=2, column=1, columnspan=1, pady=10)

        back_button = ttk.Button(frame, text="Submit", command=self.submit)
        back_button.grid(row=2, column=0, columnspan=2, pady=10)

        

    def ad(self):
        frame = ttk.Frame(self.root, padding="60")
        frame.grid(row=0, column=0)
        
        self.sort_button = ttk.Button(frame, text="Sort Hash Table", command=self.sort_hash_table)
        self.sort_button.grid(row=4, column=0, columnspan=2, pady=10)

        self.graph_button = ttk.Button(frame, text="Visualize Graph", command=self.visualize_graph)
        self.graph_button.grid(row=5, column=0, columnspan=2, pady=10)

        self.hash_table_display = tk.Text(frame, width=100, height=10)
        self.hash_table_display.grid(row=3, column=0, columnspan=2)

        self.vote_count_display = tk.Text(frame, width=100, height=5)
        self.vote_count_display.grid(row=6, column=0, columnspan=2)

        back_button = ttk.Button(frame, text="Back", command=adhome)
        back_button.grid(row=2, column=0, columnspan=2, pady=10)

        
    def submit(self):
        name = username
    
        key = int(self.preference_entry.get())
        
        value = self.voter_id_entry.get()

        if key == -1:
            self.root.quit()
            return

        user_index = key - 1
        if user_index < 0 or user_index >= self.size:
            print(f"Invalid preference. Please enter a value between 1 and {self.size}.")
        else:
            self.hash_table.insert(key, value, user_index, name)
        
    def sort_hash_table(self):
           self.hash_table.sort_table()
           self.update_hash_table_display()
           self.update_vote_count_display()

    def update_hash_table_display(self):
        self.hash_table_display.delete('1.0', tk.END)
        output = ""
        for i in range(self.size):
            output += f"Bucket {i} ({self.hash_table.get_party_name(i+1)}): {self.hash_table.table[i]}\n"
        self.hash_table_display.insert(tk.END, output)
    def update_vote_count_display(self):
        vote_counts = self.hash_table.fetch_vote_counts()
        self.vote_count_display.delete('1.0', tk.END)
        output = "Vote Counts:\n"
        for party, count in vote_counts.items():
            output += f"{party}: {count}\n"
        self.vote_count_display.insert(tk.END, output)
    def visualize_graph(self):
          self.hash_table.plot_graph()

def vote():
    for widget in root.winfo_children():
        widget.destroy()

    w2 = Frame(root, width=2000, height=700, bg='pink')
    w2.place(x=0, y=0)
    
    Button(w2, width=20, pady=7,text='Home', bg='light grey', font=('Times New Roman', 10, 'bold'), fg='black',border=0,command=home).place(x=300, y=220)

    app = App(w2)
    
def aboutus():
    for widget in root.winfo_children():
        widget.destroy()

    w6 = Frame(root, width=2000, height=500, bg='white')
    w6.place(x=0, y=0)
    

    frame = Frame(w6, width=42000, height=4500, bg='white')
    frame.place(x=-100, y=-100)
    img = PhotoImage(file='home.png',width=1000,height=700)
    Button(w6,width=800,height=900,image=img,bg='white', border=0,command=home).place(x=250,y=-120)

    about_text = (
        'About Us \n'
        'Welcome to VoteSmart! \n'
        'At VoteSmart, we are committed to transforming the \n'
        'voting experience with cutting-edge technology and a user-centric approach.\n'
        'Our mission is to make the voting process secure, transparent, and accessible for everyone.\n'
        'We believe that every vote counts, and our software ensures \n'
        'that every voice is heard.'
    )
    a = Label(w6, text=about_text, fg='black', bg='white', font=('Times New Roman', 17, 'bold'))
    a.place(x=10, y=200)
    Button(w6, width=20, pady=7, text='Home', bg='white', font=('Times New Roman', 10, 'bold'), fg='black', border=3, command=home).place(x=100, y=20)
    root.mainloop()
    
def home():
    for widget in root.winfo_children():
        widget.destroy()
        
    w1 = Frame(root, width=2000, height=500, bg='white')
    w1.place(x=0, y=0)
    
    frame = Frame(w1, width=42000, height=4500, bg='white')
    frame.place(x=-100, y=-100)
    img = PhotoImage(file='home.png',width=1000,height=700)
    Button(w1,width=800,height=900,image=img,bg='white', border=0,command=aboutus).place(x=250,y=70)

    Button(frame, width=20, pady=7, text='Log out', bg='white', font=('Times New Roman', 10, 'bold'), fg='black', border=3, command=logout).place(x=750, y=120)
    Button(frame, width=20, pady=7, text='Profile', bg='white', font=('Times New Roman', 10, 'bold'), fg='black', border=3, command=pro).place(x=150, y=120)
    Button(frame, width=20, pady=7, text='Vote', bg='white', font=('Times New Roman', 10, 'bold'), fg='black', border=3,command=vote).place(x=300, y=120)
    Button(frame, width=20, pady=7, text='Location', bg='white', font=('Times New Roman', 10, 'bold'), fg='black', border=3, command=dist).place(x=450, y=120)
    Button(frame, width=20, pady=7, text='About Us', bg='white', font=('Times New Roman', 10, 'bold'), fg='black', border=3, command=aboutus).place(x=600, y=120)
    root.mainloop()
    

def connect_to_database():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="gugan",
            database="hi"
        )
        return connection
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

def get_alphabet_count(connection):
    if connection.is_connected():
        cursor = connection.cursor()
        
        sql_query = "SELECT username FROM  users"
        print(sql_query)
        cursor.execute(sql_query)

        usernames = cursor.fetchall()

        alphabet_count = {}

        for username in usernames:
            first_alphabet = username[0][0].lower()  
            if first_alphabet.isalpha():  
                alphabet_count[first_alphabet] = alphabet_count.get(first_alphabet, 0) + 1

        cursor.close()
        return alphabet_count
    else:
        print("Failed to connect to the database.")
        return None

def knapsack_greedy(items, capacity):
    items.sort(key=lambda x: x[1], reverse=True)  

    knapsack_contents = []
    total_weight = 0
    total_value = 0

    for item in items:
        if total_weight + item[1] <= capacity:
            knapsack_contents.append(item)
            total_weight += item[1]
            total_value += item[1] 

    return knapsack_contents, total_value

def merge(left, right):
    merged = []
    while left and right:
        if left[0][1] > right[0][1]:
            merged.append(left.pop(0))
        else:
            merged.append(right.pop(0))
    merged.extend(left)
    merged.extend(right)
    return merged

def find_highest_values(contents):
    if len(contents) <= 1:
        return contents
    mid = len(contents) // 2
    left = find_highest_values(contents[:mid])
    right = find_highest_values(contents[mid:])
    return merge(left, right)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

def display_graph(alphabet_counts):
    w8 = tk.Frame(root, width=2000, height=500, bg='pink')
    w8.place(x=0, y=0)
   
    alphabet_labels = [item[0] for item in alphabet_counts]
    counts = [item[1] for item in alphabet_counts]

    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.bar(alphabet_labels, counts)
    ax.set_xlabel('Alphabet')
    ax.set_ylabel('Count')
    ax.set_title('Knapsack Contents')
    
    canvas = FigureCanvasTkAgg(fig, master=w8)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
    toolbar = NavigationToolbar2Tk(canvas, w8)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    return fig
    
def display_highest_values(highest_values):
    for widget in root.winfo_children():
        widget.destroy()
    messagebox.showinfo("Highest Values", "\n".join([f"{item[0]}: {item[1]}" for item in highest_values]))
def back_to_login():
    w4.destroy()
    create_login_window()
#admin-------------------------------
def det():
    for widget in root.winfo_children():
        widget.destroy()
    w7 = tk.Frame(root, width=2000, height=500, bg='pink')
    w7.place(x=0, y=0)
    app = App(root)
    app.ad()
   
    
def run_analysis():
    global connection, cursor

    connection = connect_to_database()
    if connection:
        alphabet_count = get_alphabet_count(connection)
        if alphabet_count:
            alphabet_count_list = [(alphabet, count) for alphabet, count in alphabet_count.items()]

            knapsack_capacity = 27
            knapsack_contents, total_value = knapsack_greedy(alphabet_count_list, knapsack_capacity)

            graph_frame = tk.Frame(w4)
            graph_frame.pack(side=tk.LEFT, padx=10, pady=10)

            graph_figure = display_graph(knapsack_contents)
            graph_canvas = FigureCanvasTkAgg(graph_figure, master=graph_frame)
            graph_canvas.draw()
            graph_canvas.get_tk_widget().pack()
            
            totalValuelab=tk.Label(w4, text=f"The total value of the knapsack is: {total_value}")
            totalValuelab.pack()
            highest_values_label = tk.Label(w4, text="\nHighest values in the knapsack:")
            highest_values_label.pack()
            highest_values = find_highest_values(knapsack_contents)
            for item in highest_values:
                label = tk.Label(w4, text=str(item))
                label.pack()

            back_button = tk.Button(w4, text="Back to Login", command=adhome)
            back_button.pack()
            
            connection.close()
def adprof():
    global logged_in_user

    for widget in root.winfo_children():
        widget.destroy()

    p3 = Frame(root, width=2000, height=500, bg='light blue')
    p3.place(x=0, y=0)

    heading = Label(p3, text='Profile', fg='black', bg='light blue', font=('Times New Roman', 23, 'bold'))
    heading.place(x=300, y=20)

    if logged_in_user:
        user_info = f"Username: {logged_in_user[1]}\n\nPassword: {logged_in_user[2]}\n\nGender: {logged_in_user[3]}"
        user_info_label = Label(p3, text=user_info, fg='black', bg='light blue', font=('Times New Roman', 14))
        user_info_label.place(x=200, y=80)
    
    Button(p3, width=20, pady=7, text='Log out', bg='brown', font=('Times New Roman', 10, 'bold'), fg='white', border=3, command=logout).place(x=220, y=250)
    Button(p3, width=20, pady=7, text='Home', bg='brown', font=('Times New Roman', 10, 'bold'), fg='white', border=3, command=adhome).place(x=220, y=300)
    img = PhotoImage(file='prof.png',width=1000,height=700)
    Label(p3,width=1000,height=700,image=img, bg='light blue').place(x=500,y=150)
    root.mainloop()
def adhome():
    global w4
    for widget in root.winfo_children():
        widget.destroy()

    w4 = Frame(root, width=2000, height=500, bg='white')
    w4.place(x=0, y=0)
    img = PhotoImage(file='home.png')
     

    
    label = Label(w4, image=img, bg='white')
    label.image = img  
    label.place(x=200, y=100)
    Button(w4, width=20, pady=7, text='Analysis', bg='white', font=('Times New Roman', 10, 'bold'), fg='black', border=3, command=run_analysis).place(x=150, y=50)
    Button(w4, width=20, pady=7, text='Total analysis', bg='white', font=('Times New Roman', 10, 'bold'), fg='black', border=3, command=det).place(x=300, y=50)
    Button(w4, width=20, pady=7, text='Profile', bg='white', font=('Times New Roman', 10, 'bold'), fg='black', border=3, command=adprof).place(x=450, y=50)
    Button(w4, width=20, pady=7, text='Log out', bg='white', font=('Times New Roman', 10, 'bold'), fg='black', border=3, command=create_login_screen).place(x=590, y=50)

    
#admin------------------------------------
def login():
    global logged_in_user
    global username
    
    username = user.get()
    password = code.get()
    gender = gen.get()
    
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="gugan",
            database="hi"
        )
        
        cursor = connection.cursor()

        cursor.execute("SELECT * FROM user WHERE username=%s AND password=%s AND gender=%s", (username, password, gender))
        
        user_data = cursor.fetchone()

        if user_data:
            logged_in_user = user_data
            login_status.config(text="Login successful!")
            if username == "vishnu" or username=="gugan" or username=="admin":
                login_status.config(text="Admin login successful!")
                adhome()
            else:
                home()
        else:
            login_status.config(text="Invalid username, password, or gender")
        
        cursor.close()
        
        connection.commit()
        connection.close()
        
    except mysql.connector.Error as err:
        login_status.config(text=f"Error: {err}")
def signup():
    username = user.get()
    password = code.get()
    gender = gen.get()

    if not username or not password or not gender:
        signup_status.config(text="Username, password, and gender cannot be empty")
        return

    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="gugan",
            database="hi"
        )
        
        cursor = connection.cursor()

        cursor.execute("SELECT * FROM user WHERE username=%s", (username,))
        existing_user = cursor.fetchone()

        if existing_user:
            signup_status.config(text="Username already exists")
        else:
            cursor.execute("INSERT INTO user (username, password, gender) VALUES (%s, %s, %s)", (username, password, gender))
            connection.commit()
            signup_status.config(text="Signup successful!")
            
    except mysql.connector.Error as err:
        signup_status.config(text=f"Error: {err}")
#fp---------------------------------
def fp_on_enter(e):
    fp_user.delete(0, 'end')

def fp_on_leave(e):
    name = fp_user.get()
    if name == '':
        fp_user.insert(0, 'Username')
#user end-------------------------
#gender---------------------------
def fp_on_enter_gen(e):
    fp_gen.delete(0, 'end')

def fp_on_leave_gen(e):
    name = fp_gen.get()
    if name == '':
        fp_gen.insert(0, 'Gender')
#gen----------------------------
def fp_on_enter_code(e):
    fp_code.delete(0, 'end')

def fp_on_leave_code(e):
    name = fp_code.get()
    if name == '':
        fp_code.insert(0, 'Password')
#--------------------------------===========================================log
def on_enter(e):
    user.delete(0, 'end')

def on_leave(e):
    name = user.get()
    if name == '':
        user.insert(0, 'Username')
#user end-------------------------
#gender---------------------------
def on_enter_gen(e):
    gen.delete(0, 'end')

def on_leave_gen(e):
    name = gen.get()
    if name == '':
        gen.insert(0, 'Gender')
#gen----------------------------
def on_enter_code(e):
    code.delete(0, 'end')

def on_leave_code(e):
    name = code.get()
    if name == '':
        code.insert(0, 'Password')
#code----------------------------
def submit_new_password():
      username = fp_user.get()
      gender = fp_gen.get()
      new_password = fp_code.get()
      if not username or not gender or not new_password:
          fp_status.config(text="All fields are required")
          return

      try:
          cursor.execute("SELECT * FROM user WHERE username=%s AND gender=%s", (username, gender))
          user_data = cursor.fetchone()

          if user_data:
               cursor.execute("UPDATE user SET password=%s WHERE username=%s AND gender=%s", (new_password, username, gender))
               connection.commit()
               fp_status.config(text="Password reset successful!")
          else:
               fp_status.config(text="Invalid username or gender")
      except mysql.connector.Error as err:
          fp_status.config(text=f"Error: {err}")
def fpass():
    global fp_user,fp_code,fp_gen,fp_status
    for widget in root.winfo_children():
        widget.destroy()

    fp1 = Frame(root, width=2000, height=500, bg='white')
    fp1.place(x=0, y=0)
    heading = Label(fp1, text='Your info', fg='black', bg='white', font=('Times New Roman', 23, 'bold'))
    heading.place(x=450, y=5)
    fp_user = Entry(fp1, width=30, fg='black', border=2, bg='lightblue', font=('Times New Roman', 14, 'bold'))
    fp_user.place(x=350, y=80)
    fp_user.insert(0, 'Username')
    fp_user.bind('<FocusIn>', fp_on_enter)
    fp_user.bind('<FocusOut>', fp_on_leave)

    
    fp_code = Entry(fp1, width=30, fg='black', border=2, bg='lightblue', font=('Times New Roman', 14, 'bold'))
    fp_code.place(x=350, y=135 )
    fp_code.insert(0, 'Password')
    fp_code.bind('<FocusIn>', fp_on_enter_code)
    fp_code.bind('<FocusOut>', fp_on_leave_code)
    
    fp_gen = Entry(fp1, width=30, fg='black', border=2, bg='lightblue', font=('Times New Roman', 14, 'bold'))
    fp_gen.place(x=350, y=190)
    fp_gen.insert(0, 'Gender')
    fp_gen.bind('<FocusIn>', fp_on_enter_gen)
    fp_gen.bind('<FocusOut>',fp_on_leave_gen)
    
    Button(fp1, width=29, pady=7, text='Change password ', bg='#57a1f8', font=('Times New Roman', 10, 'bold'), fg='black', border=3,command=submit_new_password).place(x=350, y=300)
    fp_status = Label(fp1, text='', fg='red', bg='white', font=('Times New Roman', 10, 'bold'))
    fp_status.place(x=350, y=350)
    Button(fp1, width=29, pady=7, text='Back to Log in', bg='#57a1f8', font=('Times New Roman', 10, 'bold'), fg='black', border=3, command=create_login_screen).place(x=350, y=250)

def create_login_screen():
    global user, code,gen, login_status, signup_status
    for widget in root.winfo_children():
        widget.destroy()
    img = PhotoImage(file='images.png',width=1000,height=700)
    Label(root,width=1000,height=700,image=img,bg='white').place(x=150,y=150)
    
    frame = Frame(root, width=320, height=450, bg='brown')
    frame.place(x=480, y=70)
    heading = Label(frame, text='Sign in', fg='black', bg='brown', font=('Times New Roman', 23, 'bold'))
    heading.place(x=100, y=5)
    
    user = Entry(frame, width=20, fg='black', border=2, bg='lightblue', font=('Times New Roman', 14, 'bold'))
    user.place(x=30, y=80)
    user.insert(0, 'Username')
    user.bind('<FocusIn>', on_enter)
    user.bind('<FocusOut>', on_leave)

    code = Entry(frame, width=20, fg='black', border=2, bg='lightblue', font=('Times New Roman', 14, 'bold'))
    code.place(x=30, y=140)
    code.insert(0, 'Password')
    code.bind('<FocusIn>', on_enter_code)
    code.bind('<FocusOut>', on_leave_code)
    
    gen = Entry(frame, width=20, fg='black', border=2, bg='lightblue', font=('Times New Roman', 14, 'bold'))
    gen.place(x=30, y=190)
    gen.insert(0, 'Gender')
    gen.bind('<FocusIn>', on_enter_gen)
    gen.bind('<FocusOut>', on_leave_gen)
    Frame(frame, width=295, height=2, bg='black').place(x=25, y=107)
    Frame(frame, width=295, height=2, bg='black').place(x=25, y=167)
    Frame(frame, width=295, height=2, bg='black').place(x=25, y=218)
    
    Button(frame, width=15, pady=7, text='Log in', bg='#57a1f8', font=('Times New Roman', 10, 'bold'), fg='black', border=3, command=login).place(x=50, y=250)
    
    Button(frame, width=15, pady=7, text='Sign up', bg='#57a1f8', font=('Times New Roman', 10, 'bold'), fg='black', border=3, command=signup).place(x=170, y=250)
    Button(frame, width=15, pady=7, text='Forget password', bg='#57a1f8', font=('Times New Roman', 10, 'bold'), fg='black', border=3,command=fpass).place(x=120, y=300)

    login_status = Label(frame, text='', fg='pink', bg='brown', font=('Times New Roman', 10, 'bold'))
    login_status.place(x=30, y=330)

    signup_status = Label(frame, text='', fg='pink', bg='brown', font=('Times New Roman', 10, 'bold'))
    signup_status.place(x=30, y=350)
    root.mainloop()

create_login_screen()

root.mainloop()
