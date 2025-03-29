import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./assets/Navbar";
import Footer from "./assets/footer";
import Main from "./assets/Main";
import ContentCard from "./assets/ContentCard";
import About from "./pages/About";

import "./App.css";

function App() {
 
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route
          path="/"
          element={
            <Main>
              <div className="title">
                <img id="titleImg" src="/titleBg.png" alt="Title" />
              </div>
              
              <div className="content">
  
               
              </div>
            </Main>
          }
        />
        <Route path="/about" element={<About />} />
        {/* <Route path="/chat" element={<Chat />} />
        <Route path="/signin" element={<SignIn />} /> */}
      </Routes>

      <Footer />
    </Router>
  );
}

export default App;
