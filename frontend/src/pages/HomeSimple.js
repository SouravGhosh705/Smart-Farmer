import React from "react";
import Header from "../components/Header";
import { Container, Col, Row, Card, CardBody, CardTitle, Button } from "reactstrap";
import { Link } from "react-router-dom";
import "./css/Home.css";
import HomeCards from "../components/HomeCards";

const HomeSimple = () => {
	// Feature cards data - no translations to avoid loading issues
	const features = [
		{
			title: "Crop Recommendation",
			description: "Get AI-powered crop recommendations based on soil and weather conditions",
			icon: "🌾",
			link: "/crop-recommendation",
			color: "#2e7d32"
		},
		{
			title: "Yield Prediction", 
			description: "Predict crop yields using machine learning models",
			icon: "📈",
			link: "/yield-finder",
			color: "#1976d2"
		},
		{
			title: "Price Finder",
			description: "Find current market prices and price forecasts",
			icon: "💰",
			link: "/price-finder", 
			color: "#f57c00"
		},
		{
			title: "AI Crop Doctor",
			description: "AI-powered plant disease and pest detection from images",
			icon: "🔬",
			link: "/ai-crop-doctor",
			color: "#d32f2f"
		},
		{
			title: "Multilingual Demo",
			description: "Experience our multilingual AI farming assistant",
			icon: "🌍",
			link: "/multilingual-demo",
			color: "#7b1fa2"
		}
	];

	return (
		<>
			<Header
				title='Smart Solutions for Smarter Farming!'
				desc1='AI-powered agricultural insights for modern farmers'
				desc2='Leverage machine learning for better crop decisions'
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
							About Us
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
							Tools We Offer
						</h1>
					</Col>
				</Row>

				{/* All Feature Cards in a single responsive grid */}
				<Row style={{ marginTop: '40px' }}>
					{features.map((feature, index) => (
						<Col md={6} lg={4} xl={3} key={index} style={{ marginBottom: '30px' }}>
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
											Get Started →
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
								🌍 Multilingual Support
							</h3>
							<p style={{ fontSize: '18px', color: '#666', marginBottom: '25px' }}>
								Available in 8 Indian languages for better accessibility
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
										Learn More →
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

export default HomeSimple;
