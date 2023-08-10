import './App.css';
import { Typography } from '@mui/material';
import { ConfigForm } from './components/ConfigForm';

function App() {
  return (
    <div className="App">
      <Typography variant='h2' gutterBottom>EV-ecosim Demo</Typography>
      <ConfigForm />
    </div>
  );
}

export default App;
