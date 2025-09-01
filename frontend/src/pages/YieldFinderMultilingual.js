import React, { useState } from "react";
import Header from "../components/Header";
import {
	Container,
	Row,
	Col,
	Button,
	Card,
	FormGroup,
	Label,
	Input,
} from "reactstrap";
import { Form, Spinner } from "react-bootstrap";
import API from "../utills/Api";
import ReactApexChart from "react-apexcharts";
import CropYieldDropDownOption from "../jsondropdown/crop_name_for_yield.json";
import SeasonData from "../jsondropdown/season_name_for_yield.json";
import capitalizeFirstLetter from "../utills/firstletterCap";
import cityState from "../jsondropdown/city_state_name.json";
import { useLanguage } from "../contexts/LanguageContext";
import { useBatchTranslation, useApiTranslation } from "../hooks/useTranslation";

const YieldFinderMultilingual = () => {
	const { currentLanguage, formatNumber } = useLanguage();
	const { translateApiRequest } = useApiTranslation();

	// State management
	const [SelectedState, setSelectedState] = useState("gujarat");
	const [SelectedCity, setSelectedCity] = useState("");
	const [SelectedSeason, setSelectedSeason] = useState("");
	const [SelectedCrop, setSelectedCrop] = useState("");
	const [Area, setArea] = useState("");
	const [show, setShow] = useState(false);
	const [isLoading, setisLoading] = useState(false);
	const [results, setResults] = useState();
	const [YieldBarGraph, setYieldBarGraph] = useState();
	const [PieChart, setPieChart] = useState();
	const [isError, setisError] = useState(false);
	const [ErrorMessage, setErrorMessage] = useState("");

	// Multilingual text keys for translation
	const textKeys = {
		yield_prediction: "Yield Prediction",
		state: "State",
		city: "City",
		season: "Season",
		crop: "Crop",
		area: "Area",
		predict: "Predict",
		reset: "Reset",
		loading: "Loading...",
		result: "Result",
		yield_estimate: "Yield Estimate",
		production_estimate: "Production Estimate",
		forecast: "Forecast",
		kharif_season: "Kharif Season",
		rabi_season: "Rabi Season", 
		summer_season: "Summer Season",
		select_language: "Select an option",
		enter_area: "Enter area in hectares",
		error: "Error",
		success: "Success",
		try_again: "Try Again",
		analysis: "Analysis",
		comparison: "Comparison"
	};

	// Load translations
	const { translations, loading: translationsLoading } = useBatchTranslation(textKeys);

	const getCity = (stateName) => {
		let cities = cityState[stateName];
		return cities.map((city, idx) => {
			return (
				<option key={idx} value={city}>
					{capitalizeFirstLetter(city)}
				</option>
			);
		});
	};

	const states = () => {
		var keys = Object.keys(cityState);
		return keys.map((state, idx) => {
			return (
				<option key={idx} value={state}>
					{capitalizeFirstLetter(state)}
				</option>
			);
		});
	};

	const seasons = () => {
		// Get seasons array from the JSON structure
		const seasonList = SeasonData.season || ['kharif', 'rabi', 'summer'];
		return seasonList.map((season, idx) => {
			return (
				<option key={idx} value={season}>
					{translations[`${season}_season`] || capitalizeFirstLetter(season)}
				</option>
			);
		});
	};

	const crops = () => {
		// Get crops array from the JSON structure
		const cropList = CropYieldDropDownOption.crop_name || [];
		return cropList.map((crop, idx) => {
			return (
				<option key={idx} value={crop}>
					{capitalizeFirstLetter(crop)}
				</option>
			);
		});
	};

	const showResult = async () => {
		// Clear previous error
		setisError(false);
		setErrorMessage("");
		
		// Validate form inputs
		if (!SelectedState || SelectedState === "") {
			setisError(true);
			setErrorMessage(translations.error || "Please select a state.");
			return;
		}
		
		if (!SelectedCity || SelectedCity === "") {
			setisError(true);
			setErrorMessage(translations.error || "Please select a city.");
			return;
		}
		
		if (!SelectedSeason || SelectedSeason === "") {
			setisError(true);
			setErrorMessage(translations.error || "Please select a season.");
			return;
		}
		
		if (!SelectedCrop || SelectedCrop === "") {
			setisError(true);
			setErrorMessage(translations.error || "Please select a crop.");
			return;
		}
		
		if (!Area || Area === "" || isNaN(Area) || Number(Area) <= 0) {
			setisError(true);
			setErrorMessage(translations.error || "Please enter a valid area (positive number).");
			return;
		}
		
		setArea(Number(Area));
		
		const baseData = {
			state: SelectedState,
			city: SelectedCity,
			season: SelectedSeason,
			crop: SelectedCrop,
			area: Area,
		};

		// Add language parameter for multilingual response
		const data = translateApiRequest(baseData);
		console.log('Multilingual yield request:', data);
		
		setisLoading(true);
		
		try {
			const res = await API.post("/yield", data);
			const res_data = { ...res.data };
			console.log('Multilingual yield response:', res_data);
			
			let yield_bar = {
				series: [
					{
						name: translations.yield_estimate || "Predicted Yield",
						data: res_data.barGraphvalue,
					},
				],
				options: {
					chart: {
						type: "bar",
						height: 0,
					},
					plotOptions: {
						bar: {
							borderRadius: 6,
							horizontal: true,
						},
					},
					dataLabels: {
						enabled: false,
					},
					xaxis: {
						categories: res_data.barGraphLabel || [
							translations.kharif_season || "Kharif",
							translations.rabi_season || "Rabi", 
							translations.summer_season || "Summer"
						],
					},
					colors: ['#2e7d32'],
					title: {
						text: translations.yield_estimate || 'Yield by Season',
						style: { color: '#2e7d32' }
					}
				},
			};

			let pie_chart = {
				series: res_data.pieChartValue,
				options: {
					chart: {
						width: "100%",
						type: "pie",
					},
					labels: res_data.pieChartLabel,
					colors: ['#2e7d32', '#1976d2', '#f57c00', '#7b1fa2', '#d32f2f'],
					plotOptions: {
						pie: {
							dataLabels: {
								offset: -5,
							},
						},
					},
					dataLabels: {
						formatter(val, opts) {
							const name = opts.w.globals.labels[opts.seriesIndex];
							return [name, val.toFixed(1) + "%"];
						},
					},
					legend: {
						show: true,
						position: 'bottom'
					},
					title: {
						text: translations.comparison || 'Crop Distribution',
						style: { color: '#2e7d32' }
					}
				},
			};

			setResults(res_data);
			setPieChart(pie_chart);
			setYieldBarGraph(yield_bar);
			setShow(true);
			setisLoading(false);
		} catch (err) {
			console.log(err);
			setisError(true);
			setErrorMessage(err.message || translations.error || "An error occurred while processing your request. Please check your inputs and try again.");
			setisLoading(false);
			setShow(false);
		}
	};

	const resetForm = () => {
		setSelectedState("gujarat");
		setSelectedCity("");
		setSelectedSeason("");
		setSelectedCrop("");
		setArea("");
		setShow(false);
		setisError(false);
		setErrorMessage("");
		setResults(null);
		setYieldBarGraph(null);
		setPieChart(null);
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
				title={translations.yield_prediction || 'Find How Much You Can Grow!'}
				desc1='Predict crop yields using advanced machine learning models'
				desc2='Get accurate production estimates in your preferred language'
			/>

			<Container fluid className='contant-container'>
				<Row style={{ margin: "auto", textAlign: "center" }}>
					<Col
						style={{
							margin: "24px",
							textAlign: "center",
						}}>
						<h3 style={{ fontSize: "48px", color: "#2e7d32", fontWeight: 'bold' }}>
							{translations.yield_prediction || "Yield Finder"}
						</h3>
					</Col>
				</Row>
			</Container>

			{isLoading ? (
				<Container fluid className='contant-container'>
					<Row>
						<Col style={{ textAlign: 'center', padding: '50px' }}>
							<Spinner animation="border" variant="success" style={{ width: '3rem', height: '3rem' }} />
							<h4 style={{ marginTop: '20px', color: '#2e7d32' }}>
								{translations.loading || "Loading..."}
							</h4>
						</Col>
					</Row>
				</Container>
			) : (
				<Container fluid className='contant-container'>
					<Row>
						<Col lg={12}>
							<Card style={{ padding: "30px", backgroundColor: "#f8f9fa", border: "2px solid #2e7d32", borderRadius: "12px" }}>
								<Form>
									<Row>
										<Col md={6} lg={3}>
											<FormGroup>
												<Label style={{ fontWeight: 'bold', color: '#2e7d32' }}>
													{translations.state || "State"}
												</Label>
												<Input
													type='select'
													name='State'
													value={SelectedState}
													onChange={(e) => setSelectedState(e.target.value)}
													style={{ marginTop: "8px", borderColor: '#2e7d32' }}>
													<option value="">
														-- {translations.select_language || "select an option"} --
													</option>
													{states()}
												</Input>
											</FormGroup>
										</Col>
										<Col md={6} lg={3}>
											<FormGroup>
												<Label style={{ fontWeight: 'bold', color: '#2e7d32' }}>
													{translations.city || "City"}
												</Label>
												<Input
													type='select'
													name='city'
													onChange={(e) => setSelectedCity(e.target.value)}
													value={SelectedCity}
													style={{ marginTop: "8px", borderColor: '#2e7d32' }}>
													<option value="">
														-- {translations.select_language || "select an option"} --
													</option>
													{getCity(SelectedState)}
												</Input>
											</FormGroup>
										</Col>
										<Col md={6} lg={2}>
											<FormGroup>
												<Label style={{ fontWeight: 'bold', color: '#2e7d32' }}>
													{translations.season || "Season"}
												</Label>
												<Input
													type='select'
													name='season'
													onChange={(e) => setSelectedSeason(e.target.value)}
													value={SelectedSeason}
													style={{ marginTop: "8px", borderColor: '#2e7d32' }}>
													<option value="">
														-- {translations.select_language || "select an option"} --
													</option>
													{seasons()}
												</Input>
											</FormGroup>
										</Col>
										<Col md={6} lg={2}>
											<FormGroup>
												<Label style={{ fontWeight: 'bold', color: '#2e7d32' }}>
													{translations.crop || "Crop"}
												</Label>
												<Input
													type='select'
													name='crop'
													onChange={(e) => setSelectedCrop(e.target.value)}
													value={SelectedCrop}
													style={{ marginTop: "8px", borderColor: '#2e7d32' }}>
													<option value="">
														-- {translations.select_language || "select an option"} --
													</option>
													{crops()}
												</Input>
											</FormGroup>
										</Col>
										<Col md={6} lg={2}>
											<FormGroup>
												<Label style={{ fontWeight: 'bold', color: '#2e7d32' }}>
													{translations.area || "Area"} (हेक्टेयर/Hectares)
												</Label>
												<Input
													type='number'
													placeholder={translations.enter_area || "Enter area in hectares"}
													value={Area}
													onChange={(e) => setArea(e.target.value)}
													style={{ marginTop: "8px", borderColor: '#2e7d32' }}
												/>
											</FormGroup>
										</Col>
									</Row>

									{/* Error Display */}
									{isError && (
										<Row style={{ marginTop: "20px" }}>
											<Col>
												<div className="alert alert-danger" style={{
													textAlign: 'center',
													fontWeight: 'bold'
												}}>
													❌ {ErrorMessage}
												</div>
											</Col>
										</Row>
									)}

									<Row style={{ marginTop: "30px", textAlign: 'center' }}>
										<Col>
											<Button
												color="success"
												size="lg"
												onClick={showResult}
												disabled={!SelectedState || !SelectedCity || !SelectedSeason || !SelectedCrop || !Area}
												style={{ 
													marginRight: '15px',
													padding: '12px 30px',
													fontSize: '16px',
													fontWeight: 'bold'
												}}>
												📊 {translations.predict || "Predict"}
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
								</Form>
							</Card>
						</Col>
					</Row>
				</Container>
			)}

			{/* Results Section */}
			{show && results && (
				<Container fluid style={{ marginTop: '40px', marginBottom: '40px' }}>
					{/* Result Summary Cards */}
					<Row style={{ marginBottom: '40px' }}>
						<Col md={6}>
							<Card style={{ 
								padding: '30px', 
								textAlign: 'center',
								backgroundColor: '#e8f5e8',
								border: '2px solid #2e7d32',
								borderRadius: '12px'
							}}>
								<h4 style={{ color: '#2e7d32', marginBottom: '15px' }}>
									📈 {translations.yield_estimate || "Yield Estimate"}
								</h4>
								<h2 style={{ 
									color: '#2e7d32', 
									fontWeight: 'bold',
									fontSize: '32px'
								}}>
									{formatNumber(results.predYield)} kg/hectare
								</h2>
								<p style={{ color: '#666', marginTop: '10px' }}>
									{translations.forecast || "Based on historical data and current conditions"}
								</p>
							</Card>
						</Col>
						<Col md={6}>
							<Card style={{ 
								padding: '30px', 
								textAlign: 'center',
								backgroundColor: '#e3f2fd',
								border: '2px solid #1976d2',
								borderRadius: '12px'
							}}>
								<h4 style={{ color: '#1976d2', marginBottom: '15px' }}>
									🏭 {translations.production_estimate || "Production Estimate"}
								</h4>
								<h2 style={{ 
									color: '#1976d2', 
									fontWeight: 'bold',
									fontSize: '32px'
								}}>
									{formatNumber(results.predProduction)} kg
								</h2>
								<p style={{ color: '#666', marginTop: '10px' }}>
									{translations.area || "For"} {formatNumber(Area)} {translations.area || "hectares"}
								</p>
							</Card>
						</Col>
					</Row>

					{/* Charts Section */}
					<Row>
						{/* Bar Chart - Yield by Season */}
						<Col md={6}>
							<Card style={{ padding: '20px', height: '400px' }}>
								<h5 style={{ textAlign: 'center', color: '#2e7d32', marginBottom: '20px' }}>
									📊 {translations.analysis || "Seasonal Analysis"}
								</h5>
								{YieldBarGraph && (
									<ReactApexChart
										options={YieldBarGraph.options}
										series={YieldBarGraph.series}
										type="bar"
										height={300}
									/>
								)}
							</Card>
						</Col>

						{/* Pie Chart - Crop Distribution */}
						<Col md={6}>
							<Card style={{ padding: '20px', height: '400px' }}>
								<h5 style={{ textAlign: 'center', color: '#2e7d32', marginBottom: '20px' }}>
									🥧 {translations.comparison || "Crop Comparison"}
								</h5>
								{PieChart && (
									<ReactApexChart
										options={PieChart.options}
										series={PieChart.series}
										type="pie"
										height={300}
									/>
								)}
							</Card>
						</Col>
					</Row>

					{/* Detailed Information */}
					<Row style={{ marginTop: '40px' }}>
						<Col>
							<Card style={{ 
								padding: '30px',
								backgroundColor: '#fff3e0',
								border: '2px solid #f57c00',
								borderRadius: '12px'
							}}>
								<h4 style={{ color: '#f57c00', marginBottom: '20px', textAlign: 'center' }}>
									📋 {translations.analysis || "Detailed Analysis"}
								</h4>
								
								<Row>
									<Col md={4}>
										<div style={{ textAlign: 'center', marginBottom: '20px' }}>
											<h6 style={{ color: '#f57c00', fontWeight: 'bold' }}>
												{translations.state || "State"}
											</h6>
											<p style={{ fontSize: '16px' }}>{capitalizeFirstLetter(SelectedState)}</p>
										</div>
									</Col>
									<Col md={4}>
										<div style={{ textAlign: 'center', marginBottom: '20px' }}>
											<h6 style={{ color: '#f57c00', fontWeight: 'bold' }}>
												{translations.city || "City"}
											</h6>
											<p style={{ fontSize: '16px' }}>{capitalizeFirstLetter(SelectedCity)}</p>
										</div>
									</Col>
									<Col md={4}>
										<div style={{ textAlign: 'center', marginBottom: '20px' }}>
											<h6 style={{ color: '#f57c00', fontWeight: 'bold' }}>
												{translations.season || "Season"}
											</h6>
											<p style={{ fontSize: '16px' }}>
												{translations[`${SelectedSeason}_season`] || capitalizeFirstLetter(SelectedSeason)}
											</p>
										</div>
									</Col>
								</Row>
								
								<Row>
									<Col md={6}>
										<div style={{ textAlign: 'center', marginBottom: '20px' }}>
											<h6 style={{ color: '#f57c00', fontWeight: 'bold' }}>
												{translations.crop || "Crop"}
											</h6>
											<p style={{ fontSize: '16px' }}>{capitalizeFirstLetter(SelectedCrop)}</p>
										</div>
									</Col>
									<Col md={6}>
										<div style={{ textAlign: 'center', marginBottom: '20px' }}>
											<h6 style={{ color: '#f57c00', fontWeight: 'bold' }}>
												{translations.area || "Area"}
											</h6>
											<p style={{ fontSize: '16px' }}>{formatNumber(Area)} hectares</p>
										</div>
									</Col>
								</Row>
							</Card>
						</Col>
					</Row>
				</Container>
			)}
		</>
	);
};

export default YieldFinderMultilingual;
