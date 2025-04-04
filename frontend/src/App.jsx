import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./assets/Navbar";
import Footer from "./assets/footer";
import Main from "./assets/Main";
import About from "./pages/About";
import AboutCard from "./assets/aboutCard";
import Chat from "./pages/Chat";


import "./App.css";

function App() {
  const ameyaText = "Developed the aggregator model using LangChain and Hugging Face's Transformer Library. Also responsible for developing the frontend of this website.";
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
                <h1>Contributors</h1>
                <div class="aboutCardContainerMain">
                  <AboutCard
                    imageSrc="/Xiao-Yang Liu.jpg"
                    name="Xiao-Yang Liu"
                    contentText={ameyaText}
                    linkedIn ="https://www.linkedin.com/in/xiao-yang-liu-47b72827/"
                  />

                  <AboutCard
                    imageSrc="/ameyaBarveImg.jpg"
                    name="Ameya Barve"
                    contentText={ameyaText}
                    linkedIn ="https://www.linkedin.com/in/ameyabarve1/"
                  />
                  
                  <AboutCard
                    imageSrc="/samGarnettImg.jpg"
                    name="Samuel Garnett"
                    contentText={ameyaText}
                    linkedIn ="https://www.linkedin.com/in/samuel-garnett-25a547273/"
                  />

                  <AboutCard
                    imageSrc="/ryanLeeImg.jpg"
                    name="Ryan Lee"
                    contentText={ameyaText}
                    linkedIn ="https://www.linkedin.com/in/ryan-lee100/"
                  />
                </div>
              </div>
            </Main>
          }
        />
        <Route path="/about" element={<About />} />
        <Route path="/chat" element={<Chat />} />
        {/* <Route path="/signin" element={<SignIn />} /> */}
      </Routes>

      <Footer />
    </Router>
  );
}

export default App;
