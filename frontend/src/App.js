import "./App.css";
import { BrowserRouter as Router, Route, Switch } from "react-router-dom";
import React, { useState } from "react";
import { LanguageProvider } from "./contexts/LanguageContext";
import Home from "./pages/HomeSimple";
import PriceFinder from "./pages/PriceFinder";
import YieldFinder from "./pages/YieldFinderMultilingual";
import CropRecommendation from "./pages/CropRecommendationMultilingual";
import TrainModel from "./pages/TrainModel";
import Auth from "./pages/Auth";
import ApiDoc from "./pages/ApiDoc";
import Top5 from "./pages/Top5";
import MultilingualDemo from "./pages/MultilingualDemo";
import ChatbotPage from "./pages/ChatbotPage";
import AiCropDoctor from "./pages/AiCropDoctor";
import Chatbot from "./components/Chatbot";
import LanguageSelector from "./LanguageSelector";

function App() {
	const [isChatbotOpen, setIsChatbotOpen] = useState(false);
	const [userLocation, setUserLocation] = useState({
		city: '',
		state: ''
	});

	// Try to get user's location on app load
	React.useEffect(() => {
		// You can implement geolocation or load from localStorage
		const savedLocation = localStorage.getItem('userLocation');
		if (savedLocation) {
			setUserLocation(JSON.parse(savedLocation));
		}
	}, []);

	const toggleChatbot = () => {
		setIsChatbotOpen(!isChatbotOpen);
	};

	return (
		<LanguageProvider>
			<Router>
				<div className='App'>
					{/* Global Language Selector */}
					<LanguageSelector />
					<Switch>
						<Route exact path='/'>
							<Home />
						</Route>
						<Route exact path='/crop-recommendation'>
							<CropRecommendation />
						</Route>
						<Route exact path='/yield-finder'>
							<YieldFinder />
						</Route>
						<Route exact path='/price-finder'>
							<PriceFinder />
						</Route>
						<Route exact path='/train-model'>
							<TrainModel />
						</Route>
						<Route exact path='/auth'>
							<Auth />
						</Route>
						<Route exact path='/apidoc'>
							<ApiDoc />
						</Route>
						<Route exact path='/top-5'>
							<Top5 />
						</Route>
						<Route exact path='/multilingual-demo'>
							<MultilingualDemo />
						</Route>
						<Route exact path='/chat-assistant'>
							<ChatbotPage />
						</Route>
						<Route exact path='/ai-crop-doctor'>
							<AiCropDoctor />
						</Route>
					</Switch>
					
					{/* Floating Chatbot - Available on all pages except chat page */}
					<Route path="*">
						{window.location.pathname !== '/chat-assistant' && (
							<Chatbot 
								isOpen={isChatbotOpen}
								onToggle={toggleChatbot}
								userLocation={userLocation}
							/>
						)}
					</Route>
				</div>
			</Router>
		</LanguageProvider>
	);
}

export default App;
