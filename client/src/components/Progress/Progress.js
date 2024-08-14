import * as React from 'react'; // Import React and related hooks
import { styled } from '@mui/material/styles'; // Import the styled utility from Material-UI for custom styling
import Stack from '@mui/material/Stack'; // Import the Stack component from Material-UI for layout
import LinearProgress, { linearProgressClasses } from '@mui/material/LinearProgress'; // Import LinearProgress and its classes from Material-UI
import axios from 'axios'; // Import axios for making HTTP requests
import config from '../../config/Apiconfig'; // Import the API configuration

// Create a styled version of the LinearProgress component
const BorderLinearProgress = styled(LinearProgress)(({ theme }) => ({
  height: 10, // Set the height of the progress bar
  width: 200, // Set the width of the progress bar
  borderRadius: 10, // Set the border radius for rounded corners
  // Style the background color of the progress bar's container
  [`&.${linearProgressClasses.colorPrimary}`]: {
    backgroundColor: theme.palette.grey[theme.palette.mode === 'light' ? 200 : 800], // Light or dark mode color
  },
  // Style the color and border radius of the progress bar itself
  [`& .${linearProgressClasses.bar}`]: {
    borderRadius: 10, // Rounded corners for the bar
    backgroundColor: theme.palette.mode === 'light' ? '#874f3a' : '#E53935', // Color based on light or dark mode
  },
}));

// Progress component definition
const Progress = () => {
    const [data, setData] = React.useState(0); // State to store the progress value

    React.useEffect(() => {
        getValue(); // Call getValue function when the component mounts or when the data changes
    }, [data]);

    /**
     * getValue Function
     *
     * This function retrieves the user's progress by making an API call.
     * The progress value is then updated in the state.
     */
    const getValue = async () => {
      const token = localStorage.getItem('token'); // Get the token from local storage
      const decoded = JSON.parse(atob(token.split(".")[1])); // Decode the JWT token to extract user data
      const response = await axios.get(`${config.baseURL}/${decoded.userId}`); // Make an API call to get user data
      setData(response.data.data.progress.total); // Update the data state with the retrieved progress value
    }

  return (
    <Stack spacing={2} sx={{ flexGrow: 1 }}> {/* Stack component for layout */}
      <BorderLinearProgress variant="determinate" value={data} /> {/* Custom styled progress bar with the retrieved value */}
    </Stack>
  );
}

export default Progress; // Export the Progress component
