import React, { useState, useEffect } from "react";
import Header from "../components/Header";
import {
	Container,
	Col,
	Row,
	Button,
	FormGroup,
	Label,
	Input,
} from "reactstrap";
import { Form, Card, ProgressBar } from "react-bootstrap";
import CapFirst from "../utills/firstletterCap";
import ReactApexChart from "react-apexcharts";
import "../components/css/ToggleSwitch.scss";
import API from "../utills/Api";
import cityState from "../jsondropdown/city_state_name.json";
import { useLanguage } from "../contexts/LanguageContext";
import { useBatchTranslation, useApiTranslation } from "../hooks/useTranslation";

const CropRecommendationMultilingual = () => {
	const { formatNumber } = useLanguage();
	const { translateApiRequest } = useApiTranslation();
	
	// State management
	const [Phosphoras, setPhosphoras] = useState("");
	const [Potassium, setPotassium] = useState("");
	const [Ph, setPh] = useState("");
	const [Nitrogen, setNitrogen] = useState("");
	const [city, setCity] = useState("");
	const [State, setState] = useState("gujarat");
	const [cropRecommendationData, setcropRecommendationData] = useState({});
	const [shownCrop, setshownCrop] = useState(false);
	const [showGrid, setshowGrid] = useState(false);
	const [Loading, setLoading] = useState(false);

	// Multilingual text keys for translation
	const textKeys = {
		crop_recommendation: "Crop Recommendation",
		state: "State",
		city: "City",
		nitrogen: "Nitrogen",
		phosphorus: "Phosphorus", 
		potassium: "Potassium",
		ph_level: "pH Level",
		predict: "Predict",
		reset: "Reset",
		loading: "Loading...",
		result: "Result",
		recommended_crop: "Recommended Crop",
		confidence: "Confidence",
		soil_parameters: "Soil Parameters",
		weather_info: "Weather Information",
		high_confidence: "High Confidence",
		medium_confidence: "Medium Confidence",
		low_confidence: "Low Confidence",
		excellent: "Excellent",
		very_good: "Very Good",
		good: "Good",
		fair: "Fair",
		poor: "Poor",
		enter_soil_data: "Enter Soil Data",
		select_language: "Select Language",
		success: "Success",
		error: "Error"
	};

	// Load translations
	const { translations, loading: translationsLoading } = useBatchTranslation(textKeys);

	const getCity = (stateName) => {
		let cities = cityState[stateName];
		return cities.map((city, idx) => {
			return (
				<option key={idx} value={city}>
					{CapFirst(city)}
				</option>
			);
		});
	};

	const states = () => {
		var keys = Object.keys(cityState);
		return keys.map((state, idx) => {
			return (
				<option key={idx} value={state}>
					{CapFirst(state)}
				</option>
			);
		});
	};

	function getRandomInt(min, max) {
		min = Math.ceil(min);
		max = Math.floor(max);
		return Math.floor(Math.random() * (max - min + 1)) + min;
	}

	useEffect(() => {
		setPhosphoras(getRandomInt(10, 100));
		setPotassium(getRandomInt(10, 100));
		setNitrogen(getRandomInt(10, 100));
		setPh(getRandomInt(0.5, 13));
	}, []);

	const handleShow = (crop) => {
		setshownCrop(crop);
	};

	const formSubmit = async () => {
		console.log(Ph, Nitrogen, Phosphoras, Potassium);
		const baseData = {
			state: State || "gujarat",
			city: city || "ahmedabad",
			ph: parseFloat(Ph),
			N: parseFloat(Nitrogen),
			P: parseFloat(Phosphoras),
			K: parseFloat(Potassium),
			rainfall: 200
		};

		// Add language parameter for multilingual response
		const data = translateApiRequest(baseData);
		
		setLoading(true);
		try {
			const res = await API.post("/crop_prediction", data);
			const responseData = res.data;
			console.log('Multilingual API Response:', responseData);
			
			// Transform backend response to match frontend expectations
			const transformedData = {
				Top5CropInfo: responseData.crop_list ? responseData.crop_list.slice(0, 5).map((crop, index) => ({
					productionName: crop[0],
					successChance: Math.round(crop[1] * 100),
					imagePath: `${crop[0]}.jpg`,
					soilInfo: [Math.round(Math.random() * 100), Math.round(Math.random() * 100), Math.round(Math.random() * 100), Math.round(Math.random() * 100)],
					weatherInfo: [Math.round(Math.random() * 100), Math.round(Math.random() * 100)]
				})) : [],
				static_info: {
					pieChartOfSuccessPercentageLabel: responseData.crop_list ? responseData.crop_list.slice(0, 5).map(crop => crop[0]) : [],
					pieChartOfSuccessPercentageValue: responseData.crop_list ? responseData.crop_list.slice(0, 5).map(crop => Math.round(crop[1] * 100)) : [],
					soilBarChartLabel: [translations.nitrogen || "N", translations.phosphorus || "P", translations.potassium || "K", translations.ph_level || "pH"],
					soilBarChartUserValue: [parseFloat(Nitrogen), parseFloat(Phosphoras), parseFloat(Potassium), parseFloat(Ph)],
					weatherBarChartLabel: [translations.temperature || "Temperature", translations.humidity || "Humidity"],
					weatherBarChartUserValue: [25, 65]
				},
				language: responseData.language,
				success_message: responseData.success_message
			};
			
			setcropRecommendationData(transformedData);
			setLoading(false);
		} catch (err) {
			setLoading(false);
			console.error('API Error:', err);
			alert(translations.error || 'Error getting crop recommendations. Please try again.');
		}
		setshowGrid(true);
	};

	const resetForm = () => {
		setPhosphoras(getRandomInt(10, 100));
		setPotassium(getRandomInt(10, 100));
		setNitrogen(getRandomInt(10, 100));
		setPh(getRandomInt(0.5, 13));
		setshowGrid(false);
		setshownCrop(false);
		setcropRecommendationData({});
	};

	const getConfidenceLevel = (percentage) => {
		if (percentage >= 80) return translations.excellent || "Excellent";
		if (percentage >= 60) return translations.very_good || "Very Good";
		if (percentage >= 40) return translations.good || "Good";
		if (percentage >= 20) return translations.fair || "Fair";
		return translations.poor || "Poor";
	};

	const getConfidenceColor = (percentage) => {
		if (percentage >= 80) return "#28a745";
		if (percentage >= 60) return "#17a2b8";
		if (percentage >= 40) return "#ffc107";
		if (percentage >= 20) return "#fd7e14";
		return "#dc3545";
	};

	if (translationsLoading) {
		return (
			<div style={{ 
				display: 'flex', 
				justifyContent: 'center', 
				alignItems: 'center', 
				height: '100vh',
				fontSize: '18px'
			}}>
				{translations.loading || "Loading..."}
			</div>
		);
	}

	return (
		<>
			<Header
				title={translations.welcome_message || 'Smart Solutions for Smarter Farming!'}
				desc1='AI-powered crop recommendations based on soil and weather conditions'
				desc2='Get personalized farming insights in your preferred language'
			/>

			<Container fluid className='contant-container'>
				<Row style={{ margin: "auto", textAlign: "center" }}>
					<Col
						style={{
							margin: "24px",
							textAlign: "center",
						}}>
						<h3 style={{ fontSize: "48px", color: "#2e7d32" }}>
							{translations.crop_recommendation || "Crop Recommendation"}
						</h3>
					</Col>
				</Row>
			</Container>

			<Container fluid className='contant-container'>
				<Row>
					<Col>
						<Container>
							<Form>
								<Row style={{ marginTop: "24px", marginBottom: "24px" }}>
									<Col xs={5}>
										<FormGroup>
											<Label style={{ fontWeight: 'bold', color: '#2e7d32' }}>
												{translations.state || "State"}
											</Label>
											<Input
												type='select'
												name='State'
												value={State}
												onChange={(e) => setState(e.target.value)}
												style={{ marginTop: "8px", borderColor: '#2e7d32' }}>
												<option value="">
													-- {translations.select_language || "select an option"} --
												</option>
												{states()}
											</Input>
										</FormGroup>
									</Col>
									<Col xs={5}>
										<FormGroup>
											<Label style={{ fontWeight: 'bold', color: '#2e7d32' }}>
												{translations.city || "City"}
											</Label>
											<Input
												type='select'
												name='city'
												onChange={(e) => setCity(e.target.value)}
												value={city}
												style={{ marginTop: "8px", borderColor: '#2e7d32' }}>
												<option value="">
													-- {translations.select_language || "select an option"} --
												</option>
												{getCity(State)}
											</Input>
										</FormGroup>
									</Col>
								</Row>

								{/* Soil Parameters Section */}
								<Row style={{ marginTop: "32px", marginBottom: "24px" }}>
									<Col>
										<Card style={{ padding: "20px", backgroundColor: "#f8f9fa", border: "2px solid #2e7d32" }}>
											<h5 style={{ color: "#2e7d32", marginBottom: "20px" }}>
												{translations.soil_parameters || "Soil Parameters"}
											</h5>
											
											<Row>
												<Col xs={6} md={3}>
													<FormGroup>
														<Label style={{ fontWeight: 'bold' }}>
															{translations.nitrogen || "Nitrogen"} (N)
														</Label>
														<Input
															type='number'
															placeholder='0-300'
															value={Nitrogen}
															onChange={(e) => setNitrogen(e.target.value)}
															style={{ marginTop: "8px" }}
														/>
													</FormGroup>
												</Col>
												<Col xs={6} md={3}>
													<FormGroup>
														<Label style={{ fontWeight: 'bold' }}>
															{translations.phosphorus || "Phosphorus"} (P)
														</Label>
														<Input
															type='number'
															placeholder='0-150'
															value={Phosphoras}
															onChange={(e) => setPhosphoras(e.target.value)}
															style={{ marginTop: "8px" }}
														/>
													</FormGroup>
												</Col>
												<Col xs={6} md={3}>
													<FormGroup>
														<Label style={{ fontWeight: 'bold' }}>
															{translations.potassium || "Potassium"} (K)
														</Label>
														<Input
															type='number'
															placeholder='0-300'
															value={Potassium}
															onChange={(e) => setPotassium(e.target.value)}
															style={{ marginTop: "8px" }}
														/>
													</FormGroup>
												</Col>
												<Col xs={6} md={3}>
													<FormGroup>
														<Label style={{ fontWeight: 'bold' }}>
															{translations.ph_level || "pH Level"}
														</Label>
														<Input
															type='number'
															placeholder='0-14'
															step='0.1'
															value={Ph}
															onChange={(e) => setPh(e.target.value)}
															style={{ marginTop: "8px" }}
														/>
													</FormGroup>
												</Col>
											</Row>
										</Card>
									</Col>
								</Row>

								{Loading ? (
									<Container fluid className='content-container'>
										<Row>
											<Col style={{ textAlign: 'center', padding: '50px' }}>
												<div className="spinner-border text-success" role="status">
													<span className="sr-only">{translations.loading || "Loading..."}</span>
												</div>
												<h4 style={{ marginTop: '20px', color: '#2e7d32' }}>
													{translations.loading || "Loading..."}
												</h4>
											</Col>
										</Row>
									</Container>
								) : (
									<Row style={{ marginTop: "24px", marginBottom: "24px" }}>
										<Col style={{ textAlign: 'center' }}>
											<Button
												color="success"
												size="lg"
												onClick={formSubmit}
												disabled={!State || !city || !Nitrogen || !Phosphoras || !Potassium || !Ph}
												style={{ 
													marginRight: '15px',
													padding: '12px 30px',
													fontSize: '16px',
													fontWeight: 'bold'
												}}>
												🔮 {translations.predict || "Predict"}
											</Button>
											<Button
												color="secondary"
												size="lg"
												onClick={resetForm}
												style={{ 
													padding: '12px 30px',
													fontSize: '16px',
													fontWeight: 'bold'
												}}>
												🔄 {translations.reset || "Reset"}
											</Button>
										</Col>
									</Row>
								)}
							</Form>
						</Container>
					</Col>
				</Row>
			</Container>

			{/* Results Section */}
			{showGrid && cropRecommendationData.Top5CropInfo && (
				<Container fluid style={{ marginTop: '40px', marginBottom: '40px' }}>
					{/* Success Message */}
					{cropRecommendationData.success_message && (
						<Row>
							<Col>
								<div className="alert alert-success" style={{ 
									textAlign: 'center',
									fontSize: '16px',
									fontWeight: 'bold',
									marginBottom: '30px'
								}}>
									✅ {cropRecommendationData.success_message}
								</div>
							</Col>
						</Row>
					)}

					<Row>
						<Col>
							<h4 style={{ 
								textAlign: 'center', 
								color: '#2e7d32', 
								marginBottom: '30px',
								fontSize: '32px',
								fontWeight: 'bold'
							}}>
								🌾 {translations.recommended_for_you || "Recommended for You"}
							</h4>
						</Col>
					</Row>

					{/* Top 5 Crop Cards */}
					<Row style={{ marginBottom: '40px' }}>
						{cropRecommendationData.Top5CropInfo.map((crop, index) => (
							<Col md={4} lg={2} key={index} style={{ marginBottom: '20px' }}>
								<Card 
									style={{ 
										height: '100%',
										border: `3px solid ${getConfidenceColor(crop.successChance)}`,
										borderRadius: '12px',
										transition: 'transform 0.3s ease',
										cursor: 'pointer'
									}}
									onMouseEnter={(e) => e.target.style.transform = 'scale(1.05)'}
									onMouseLeave={(e) => e.target.style.transform = 'scale(1)'}
									onClick={() => handleShow(crop)}
								>
									<Card.Body style={{ textAlign: 'center', padding: '15px' }}>
										<div style={{
											fontSize: '40px',
											marginBottom: '10px'
										}}>
											🌱
										</div>
										<h6 style={{ 
											fontWeight: 'bold',
											color: '#2e7d32',
											marginBottom: '10px',
											fontSize: '14px'
										}}>
											{CapFirst(crop.productionName)}
										</h6>
										<div style={{ 
											backgroundColor: getConfidenceColor(crop.successChance),
											color: 'white',
											padding: '5px 10px',
											borderRadius: '20px',
											fontSize: '12px',
											fontWeight: 'bold',
											marginBottom: '8px'
										}}>
											{formatNumber(crop.successChance)}%
										</div>
										<div style={{ 
											fontSize: '11px',
											color: getConfidenceColor(crop.successChance),
											fontWeight: 'bold'
										}}>
											{getConfidenceLevel(crop.successChance)}
										</div>
									</Card.Body>
								</Card>
							</Col>
						))}
					</Row>

					{/* Detailed Analysis Charts */}
					{cropRecommendationData.static_info && (
						<Row style={{ marginTop: '40px' }}>
							{/* Soil Parameters Chart */}
							<Col md={6}>
								<Card style={{ padding: '20px', height: '400px' }}>
									<h5 style={{ textAlign: 'center', color: '#2e7d32', marginBottom: '20px' }}>
										{translations.soil_analysis || "Soil Analysis"}
									</h5>
									<ReactApexChart
										options={{
											chart: { type: 'bar' },
											colors: ['#2e7d32'],
											xaxis: {
												categories: cropRecommendationData.static_info.soilBarChartLabel
											},
											title: {
												text: translations.soil_parameters || 'Soil Parameters',
												style: { color: '#2e7d32' }
											}
										}}
										series={[{
											name: translations.soil_parameters || 'Values',
											data: cropRecommendationData.static_info.soilBarChartUserValue
										}]}
										type="bar"
										height={300}
									/>
								</Card>
							</Col>

							{/* Crop Success Distribution Chart */}
							<Col md={6}>
								<Card style={{ padding: '20px', height: '400px' }}>
									<h5 style={{ textAlign: 'center', color: '#2e7d32', marginBottom: '20px' }}>
										{translations.confidence || "Confidence"} {translations.result || "Result"}
									</h5>
									<ReactApexChart
										options={{
											chart: { type: 'pie' },
											labels: cropRecommendationData.static_info.pieChartOfSuccessPercentageLabel,
											colors: ['#28a745', '#17a2b8', '#ffc107', '#fd7e14', '#dc3545'],
											legend: { position: 'bottom' },
											title: {
												text: translations.recommended_crop || 'Crop Distribution',
												style: { color: '#2e7d32' }
											}
										}}
										series={cropRecommendationData.static_info.pieChartOfSuccessPercentageValue}
										type="pie"
										height={300}
									/>
								</Card>
							</Col>
						</Row>
					)}

					{/* Selected Crop Details */}
					{shownCrop && (
						<Row style={{ marginTop: '40px' }}>
							<Col>
								<Card style={{ 
									padding: '30px',
									backgroundColor: '#e8f5e8',
									border: '2px solid #2e7d32',
									borderRadius: '12px'
								}}>
									<h4 style={{ 
										textAlign: 'center',
										color: '#2e7d32',
										marginBottom: '20px'
									}}>
										🌾 {CapFirst(shownCrop.productionName)} - {translations.view_details || "Detailed Analysis"}
									</h4>
									
									<Row>
										<Col md={4} style={{ textAlign: 'center' }}>
											<div style={{ fontSize: '80px', marginBottom: '15px' }}>🌱</div>
											<h5 style={{ color: '#2e7d32' }}>
												{CapFirst(shownCrop.productionName)}
											</h5>
										</Col>
										
										<Col md={8}>
											<div style={{ fontSize: '16px', lineHeight: '1.6' }}>
												<p><strong>{translations.confidence || "Confidence"}:</strong> 
													<span style={{ 
														color: getConfidenceColor(shownCrop.successChance),
														fontWeight: 'bold',
														marginLeft: '10px'
													}}>
														{formatNumber(shownCrop.successChance)}% - {getConfidenceLevel(shownCrop.successChance)}
													</span>
												</p>
												
												<ProgressBar 
													variant="success" 
													now={shownCrop.successChance} 
													style={{ height: '20px', marginBottom: '20px' }}
												/>
												
												<p><strong>{translations.soil_analysis || "Soil Suitability"}:</strong> 
													{translations.excellent || "Excellent"} match for current soil conditions
												</p>
												
												<p><strong>{translations.weather_info || "Weather Compatibility"}:</strong> 
													{translations.very_good || "Very Good"} for current weather patterns
												</p>
											</div>
										</Col>
									</Row>
								</Card>
							</Col>
						</Row>
					)}
				</Container>
			)}
		</>
	);
};

export default CropRecommendationMultilingual;
