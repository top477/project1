import { Router } from "express"; // Importing the Router class from express
import { 
    registerUserController,  // Importing the controller functions
    deleteUserController, 
    getByIdUserController, 
    getUserController, 
    loginUserController, 
    updateUserController 
} from "../controller/user-controller"; // Importing user-related controller functions from the user-controller file
import verifyToken from "../middleware/auth-middleware"; // Importing middleware function to verify tokens

const userRouter = Router(); // Creating a new Router instance

// Route for '/user' path
userRouter.route('/')
    .get(verifyToken, getUserController) // GET request to fetch all users, with token verification
    .post(registerUserController); // POST request to register a new user

// Route for '/user/:id' path
userRouter.route('/:id')
    .get(getByIdUserController) // GET request to fetch a user by ID
    .delete(verifyToken, deleteUserController) // DELETE request to remove a user by ID, with token verification
    .put(updateUserController); // PUT request to update a user by ID

// Route for '/user/login' path
userRouter.route('/login')
    .post(loginUserController); // POST request to log in a user

export default userRouter; // Exporting the userRouter instance for use in other parts of the application
