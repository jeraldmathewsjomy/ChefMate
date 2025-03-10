# ChefMate

**ChefMate** is a web-based Recipe Recommendation System designed to help users discover new recipes based on their preferences. Built with HTML, CSS, and Flask, ChefMate offers an intuitive interface to search and view detailed recipes.

## Deployed Link

https://chefmate-v3gs.onrender.com/
("Since Render is a free hosting service, the application may experience a delay of approximately 50 seconds when starting up.") 

## Features

- **Recipe Search**: Users can search for recipes using keywords or ingredients.
- **Recipe Recommendations**: Based on the search query, ChefMate provides a list of recommended recipes.
- **Detailed View**: Each recipe includes comprehensive details such as ingredients, preparation steps, and cooking time.

  

## 🛠️ Core Components

| Technology       | Role in the Project |
|-----------------|--------------------|
| Python (3.x)    | Primary programming language for backend logic and ML models. |
| Flask           | Web framework for handling routes, rendering templates, and managing backend logic. |
| Jinja2          | Template engine to dynamically render HTML pages with backend data. |
| Gunicorn        | WSGI server for running Flask applications in production. |
| Virtualenv      | Creates an isolated Python environment to manage dependencies. |

## 🎨 Frontend Technologies

| Technology    | Role in the Project |
|--------------|--------------------|
| HTML        | Provides the structure for web pages (forms, buttons, results display). |
| CSS         | Styles the website with modern design and responsiveness. |
| JavaScript  | Enhances user interaction and dynamic content updates. |
| Pexels API  | Fetches high-quality food images to enhance recipe displays. |
| CSV File    | Stores and loads dataset information (e.g., recipes and ingredients). |

## 🚀 Deployment Tools

| Technology | Role in the Project |
|-----------|--------------------|
| Git       | Version control to track project changes. |
| GitHub    | Hosts the project repository for collaboration and deployment. |
| Render    | Cloud hosting service used for deploying the Flask application. |

## Screenshots

**HOMEPAGE**

![Screenshot 2025-03-04 204157](https://github.com/user-attachments/assets/5bbb8823-fe4f-4fe2-afb1-de4ed2a805ba)
*Description: The Introductory Page*

**MENU**

![Screenshot 2025-03-04 204219](https://github.com/user-attachments/assets/609a5801-c7ca-42fc-8530-a81d72d0683b)
*Description: Displays shuffled recipies available*

**RECOMMENTATION SYSTEM PAGE**

![Screenshot 2025-03-04 204656](https://github.com/user-attachments/assets/1bd2a6b6-ef21-4000-ab1a-6e357737cf03)
*Description: The main interface where users can search for recipes.*

**RESULT PAGE**

![image](https://github.com/user-attachments/assets/8ac2e9df-b002-43d2-9b9f-092aeb1fe42b)
*Description: The main interface where users can search for recipes.*

**RECIPE PAGE**
![image](https://github.com/user-attachments/assets/223b3cd9-2537-41c5-b062-44b0e5e567ac)
*Description: Detailed view of a selected recipe.*

## Installation

To set up ChefMate locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jeraldmathewsjomy/ChefMate.git
   ```
2. **Navigate to the project directory**:
   ```bash
   cd ChefMate
   ```
3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Flask application**:
   ```bash
   python app.py
   ```
5. **Access the application**: Open your browser and navigate to `http://127.0.0.1:5000/`.

## Usage

1. **Search for Recipes**: Enter keywords or ingredients in the search bar on the home page.
2. **View Recommendations**: Browse through the list of recommended recipes.
3. **Get Recipe Details**: Click on a recipe to view its ingredients, preparation steps, and other details.

## Contributing

We welcome contributions to enhance ChefMate. To contribute:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Make your changes and commit them:
   ```bash
   git commit -m 'Add feature name'
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a Pull Request detailing your changes.

## 🤝 Contributing
Feel free to submit issues, fork the repository, and open pull requests to contribute to Chefmate!

## 📞 Contact
For any inquiries or collaboration, reach out via GitHub issues or email: `jeraldmathewsjomy@gmail.com`.

## Acknowledgements

- [Flask](https://flask.palletsprojects.com/) - The web framework used.
- [Jinja2](https://jinja.palletsprojects.com/) - Templating engine.
- [Bootstrap](https://getbootstrap.com/) - For responsive design components.

