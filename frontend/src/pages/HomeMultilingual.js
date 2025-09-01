import React from "react";
import Header from "../components/Header";
import { Container, Col, Row, Card, CardBody, CardTitle, Button } from "reactstrap";
import { Link } from "react-router-dom";
import "./css/Home.css";
import HomeCards from "../components/HomeCards";
import { useBatchTranslation } from "../hooks/useTranslation";

// Move textKeys outside component to prevent recreation
const TEXT_KEYS = {
	app_title: "Smart Farmer",
	welcome_message: "Welcome to Smart Farmer - Your AI Agricultural Assistant",
	about_us: "About Us",
	tools_we_offer: "Tools We Offer",
	get_started: "Get Started",
	learn_more: "Learn More",
	crop_recommendation: "Crop Recommendation",
	yield_prediction: "Yield Prediction",
	price_finder: "Price Finder",
	multilingual_demo: "Multilingual Demo",
	ai_crop_doctor: "AI Crop Doctor",
	home: "Home",
	dashboard: "Dashboard",
	ai_powered_desc: "AI-powered agricultural insights for modern farmers",
	smart_farming_desc: "Leverage machine learning for better crop decisions",
	multilingual_desc: "Available in 8 Indian languages for better accessibility",
	crop_doctor_desc: "AI-powered plant disease and pest detection from images"
};

const HomeMultilingual = () => {

	// Load translations
	const { translations, loading: translationsLoading } = useBatchTranslation(TEXT_KEYS);

	// Feature cards data
	const features = [
		{
			title: translations.crop_recommendation || "Crop Recommendation",
			description: "Get AI-powered crop recommendations based on soil and weather conditions",
			icon: "🌾",
			link: "/crop-recommendation",
			color: "#2e7d32"
		},
		{
			title: translations.yield_prediction || "Yield Prediction", 
			description: "Predict crop yields using machine learning models",
			icon: "📈",
			link: "/yield-finder",
			color: "#1976d2"
		},
		{
			title: translations.price_finder || "Price Finder",
			description: "Find current market prices and price forecasts",
			icon: "💰",
			link: "/price-finder", 
			color: "#f57c00"
		},
		{
			title: translations.ai_crop_doctor || "AI Crop Doctor",
			description: translations.crop_doctor_desc || "AI-powered plant disease and pest detection from images",
			icon: "🔬",
			link: "/ai-crop-doctor",
			color: "#d32f2f"
		},
		{
			title: translations.multilingual_demo || "Multilingual Demo",
			description: "Experience our multilingual AI farming assistant",
			icon: "🌍",
			link: "/multilingual-demo",
			color: "#7b1fa2"
		}
	];

	// Add timeout fallback for translations - don't block UI for too long
	const [showFallback, setShowFallback] = React.useState(false);
	
	React.useEffect(() => {
		const timer = setTimeout(() => {
			if (translationsLoading) {
				console.warn('Translation loading too long, showing fallback');
				setShowFallback(true);
			}
		}, 2000); // Show fallback after 2 seconds
		
		return () => clearTimeout(timer);
	}, [translationsLoading]);
	
	if (translationsLoading && !showFallback) {
		return (
			<div style={{ 
				display: 'flex', 
				justifyContent: 'center', 
				alignItems: 'center', 
				height: '100vh',
				fontSize: '18px',
				color: '#2e7d32'
			}}>
				<div style={{ textAlign: 'center' }}>
					<div style={{ fontSize: '48px', marginBottom: '20px' }}>🌾</div>
					<div>Loading Smart Farmer...</div>
				</div>
			</div>
		);
	}

	return (
		<>
			<Header
				title={translations.welcome_message || 'Smart Solutions for Smarter Farming!'}
				desc1={translations.ai_powered_desc || 'AI-powered agricultural insights for modern farmers'}
				desc2={translations.smart_farming_desc || 'Leverage machine learning for better crop decisions'}
			/>

			{/* About Us Section */}
			<Container fluid className='contant-container' style={{ backgroundColor: '#f8f9fa', padding: '60px 0' }}>
				<Row className='heading-container'>
					<Col className='heading-text-container'>
						<h1 className='heading-text' style={{ 
							fontSize: '48px',
							color: '#2e7d32',
							textAlign: 'center',
							marginBottom: '40px',
							fontWeight: 'bold'
						}}>
							{translations.about_us || "About Us"}
						</h1>
					</Col>
				</Row>
				<Row>
					<Col>
						<HomeCards aboutus={true} />
					</Col>
				</Row>
			</Container>

			{/* Features Section */}
			<Container fluid className='contant-container' style={{ padding: '60px 0' }}>
				<Row className='heading-container'>
					<Col className='heading-text-container'>
						<h1 className='heading-text' style={{ 
							fontSize: '48px',
							color: '#2e7d32',
							textAlign: 'center',
							marginBottom: '40px',
							fontWeight: 'bold'
						}}>
							{translations.tools_we_offer || "Tools We Offer"}
						</h1>
					</Col>
				</Row>

				{/* Feature Cards */}
				<Row style={{ marginTop: '40px' }}>
					{features.slice(0, 4).map((feature, index) => (
						<Col md={6} lg={3} key={index} style={{ marginBottom: '30px' }}>
							<Card style={{
								height: '100%',
								border: `3px solid ${feature.color}`,
								borderRadius: '15px',
								transition: 'all 0.3s ease',
								cursor: 'pointer',
								boxShadow: '0 4px 15px rgba(0,0,0,0.1)'
							}}
							onMouseEnter={(e) => {
								e.currentTarget.style.transform = 'translateY(-10px)';
								e.currentTarget.style.boxShadow = '0 8px 25px rgba(0,0,0,0.15)';
							}}
							onMouseLeave={(e) => {
								e.currentTarget.style.transform = 'translateY(0)';
								e.currentTarget.style.boxShadow = '0 4px 15px rgba(0,0,0,0.1)';
							}}>
								<CardBody style={{ 
									textAlign: 'center', 
									padding: '30px 20px',
									display: 'flex',
									flexDirection: 'column',
									justifyContent: 'space-between',
									height: '100%'
								}}>
									<div>
										<div style={{ 
											fontSize: '60px', 
											marginBottom: '20px',
											color: feature.color
										}}>
											{feature.icon}
										</div>
										<CardTitle style={{ 
											fontSize: '20px',
											fontWeight: 'bold',
											color: feature.color,
											marginBottom: '15px'
										}}>
											{feature.title}
										</CardTitle>
										<p style={{ 
											fontSize: '14px',
											color: '#666',
											lineHeight: '1.5',
											marginBottom: '20px'
										}}>
											{feature.description}
										</p>
									</div>
									<Link to={feature.link} style={{ textDecoration: 'none' }}>
										<Button 
											style={{
												backgroundColor: feature.color,
												border: 'none',
												padding: '10px 25px',
												fontSize: '14px',
												fontWeight: 'bold',
												borderRadius: '25px',
												width: '100%'
											}}
										>
											{translations.get_started || "Get Started"} →
										</Button>
									</Link>
								</CardBody>
							</Card>
						</Col>
					))}
				</Row>

				{/* Second Row of Features */}
				<Row style={{ marginTop: '30px' }}>
					{features.slice(3).map((feature, index) => (
						<Col md={6} lg={6} key={index + 4} style={{ marginBottom: '30px' }}>
							<Card style={{
								height: '100%',
								border: `3px solid ${feature.color}`,
								borderRadius: '15px',
								transition: 'all 0.3s ease',
								cursor: 'pointer',
								boxShadow: '0 4px 15px rgba(0,0,0,0.1)'
							}}
							onMouseEnter={(e) => {
								e.currentTarget.style.transform = 'translateY(-10px)';
								e.currentTarget.style.boxShadow = '0 8px 25px rgba(0,0,0,0.15)';
							}}
							onMouseLeave={(e) => {
								e.currentTarget.style.transform = 'translateY(0)';
								e.currentTarget.style.boxShadow = '0 4px 15px rgba(0,0,0,0.1)';
							}}>
								<CardBody style={{ 
									textAlign: 'center', 
									padding: '30px 20px',
									display: 'flex',
									flexDirection: 'column',
									justifyContent: 'space-between',
									height: '100%'
								}}>
									<div>
										<div style={{ 
											fontSize: '60px', 
											marginBottom: '20px',
											color: feature.color
										}}>
											{feature.icon}
										</div>
										<CardTitle style={{ 
											fontSize: '20px',
											fontWeight: 'bold',
											color: feature.color,
											marginBottom: '15px'
										}}>
											{feature.title}
										</CardTitle>
										<p style={{ 
											fontSize: '14px',
											color: '#666',
											lineHeight: '1.5',
											marginBottom: '20px'
										}}>
											{feature.description}
										</p>
									</div>
									<Link to={feature.link} style={{ textDecoration: 'none' }}>
										<Button 
											style={{
												backgroundColor: feature.color,
												border: 'none',
												padding: '10px 25px',
												fontSize: '14px',
												fontWeight: 'bold',
												borderRadius: '25px',
												width: '100%'
											}}
										>
											{translations.get_started || "Get Started"} →
										</Button>
									</Link>
								</CardBody>
							</Card>
						</Col>
					))}
				</Row>

				{/* Multilingual Support Highlight */}
				<Row style={{ marginTop: '60px' }}>
					<Col>
						<Card style={{
							backgroundColor: '#e8f5e8',
							border: '2px solid #2e7d32',
							borderRadius: '15px',
							padding: '30px',
							textAlign: 'center'
						}}>
							<h3 style={{ color: '#2e7d32', marginBottom: '20px', fontWeight: 'bold' }}>
								🌍 {translations.multilingual_demo || "Multilingual Support"}
							</h3>
							<p style={{ fontSize: '18px', color: '#666', marginBottom: '25px' }}>
								{translations.multilingual_desc || "Available in 8 Indian languages for better accessibility"}
							</p>
							<Row>
								<Col md={8} style={{ margin: '0 auto' }}>
									<div style={{ 
										display: 'flex', 
										justifyContent: 'center', 
										flexWrap: 'wrap', 
										gap: '15px' 
									}}>
										{['🇮🇳 English', '🇮🇳 हिन्दी', '🇮🇳 ગુજરાતી', '🇮🇳 ਪੰਜਾਬੀ', '🇮🇳 मराठी', '🇮🇳 தமிழ்', '🇮🇳 తెలుగు', '🇮🇳 বাংলা'].map((lang, idx) => (
											<span 
												key={idx}
												style={{
													backgroundColor: '#2e7d32',
													color: 'white',
													padding: '8px 15px',
													borderRadius: '20px',
													fontSize: '14px',
													fontWeight: 'bold'
												}}
											>
												{lang}
											</span>
										))}
									</div>
								</Col>
							</Row>
							<div style={{ marginTop: '25px' }}>
								<Link to="/multilingual-demo" style={{ textDecoration: 'none' }}>
									<Button 
										color="success" 
										size="lg"
										style={{
											padding: '12px 30px',
											fontSize: '16px',
											fontWeight: 'bold',
											borderRadius: '25px'
										}}
									>
										{translations.learn_more || "Learn More"} →
									</Button>
								</Link>
							</div>
						</Card>
					</Col>
				</Row>
			</Container>

			{/* Original Tools Section */}
			<Container fluid className='contant-container' style={{ backgroundColor: '#f8f9fa', padding: '60px 0' }}>
				<Row>
					<Col>
						<HomeCards tools={true} />
					</Col>
				</Row>
			</Container>
		</>
	);
};

export default HomeMultilingual;
