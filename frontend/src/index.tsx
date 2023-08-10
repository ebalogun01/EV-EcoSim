import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import reportWebVitals from './reportWebVitals';
import { createTheme, ThemeProvider } from '@mui/material';

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);

// Create theme to customize components app-wide
const theme = createTheme({
  components: {
    MuiAccordion: {
      defaultProps: {
        disableGutters: true,
        elevation: 0,
        square: true,
      }
    },
    MuiFormHelperText: {
      defaultProps: {
        style: {textAlign: 'right'}
      }
    }
  }
})

root.render(
  <React.StrictMode>
    <ThemeProvider theme={theme}>
      <App />
    </ThemeProvider>
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
