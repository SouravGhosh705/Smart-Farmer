import React, { useState } from "react";
import Header from "../components/Header";
import {
	Container,
	Row,
	Col,
	Button,
	Card,
	CardTitle,
	CardText,
	FormGroup,
	Label,
	Input,
} from "reactstrap";
import { Form, Spinner } from "react-bootstrap";
import API from "../utills/Api";
import ReactApexChart from "react-apexcharts";
import CropYieldDropDownOption from "../jsondropdown/crop_name_for_yield.json";
import SeasonName from "../jsondropdown/season_name_for_yield.json";
import capitalizeFirstLetter from "../utills/firstletterCap";
import cityState from "../jsondropdown/city_state_name.json";

const YieldFinder = () => {
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
	const getCity = (stateName) => {
		let cities = cityState[stateName];
		console.log(cities);
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
	const showResult = async () => {
		// Clear previous error
		setisError(false);
		setErrorMessage("");
		
		// Validate form inputs
		if (!SelectedState || SelectedState === "") {
			setisError(true);
			setErrorMessage("Please select a state.");
			return;
		}
		
		if (!SelectedCity || SelectedCity === "") {
			setisError(true);
			setErrorMessage("Please select a city.");
			return;
		}
		
		if (!SelectedSeason || SelectedSeason === "") {
			setisError(true);
			setErrorMessage("Please select a season.");
			return;
		}
		
		if (!SelectedCrop || SelectedCrop === "") {
			setisError(true);
			setErrorMessage("Please select a crop.");
			return;
		}
		
		if (!Area || Area === "" || isNaN(Area) || Number(Area) <= 0) {
			setisError(true);
			setErrorMessage("Please enter a valid area (positive number).");
			return;
		}
		
		setArea(Number(Area));
		console.log(Area);
		const data = {
			state: SelectedState,
			city: SelectedCity,
			season: SelectedSeason,
			crop: SelectedCrop,
			area: Area,
		};
		console.log(data);
		setisLoading(true);
		await API.post("/yield", { ...data })
			.then((res) => {
				const res_data = { ...res.data };
				console.log(res_data);
				let yield_bar = {
					series: [
						{
							name: "Predicted Yield",
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
							categories: res_data.barGraphLabel,
						},
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
						theme: {
							monochrome: {
								enabled: true,
							},
						},
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
						},
					},
				};
				setResults(res_data);
				setPieChart(pie_chart);
				setYieldBarGraph(yield_bar);
				setShow(true);
				setisLoading(false);
			})
			.catch((err) => {
				console.log(err);
				setisError(true);
				setErrorMessage(err.message || "An error occurred while processing your request. Please check your inputs and try again.");
				setisLoading(false);
				setShow(false);
			});
	};
	return (
		<>
			<Header
				title='Find How Much You Can Grow!'
				desc1='This is a simple hero unit, a simple Jumbotron-style component
								for calling extra attention to featured content or information.'
				desc2='It uses utility classes for typography and spacing to space
                content out within the larger container.'
			/>
			<Container fluid className='contant-container'>
				<Row style={{ margin: "auto", textAlign: "center" }}>
					<Col
						style={{
							margin: "24px",
							textAlign: "center",
						}}>
						<h3 style={{ fontSize: "48px" }}>Yield Finder</h3>
					</Col>
				</Row>
			</Container>
			{isLoading ? (
				<Container fluid className='contant-container'>
					<Row>
						<Col xs={2} style={{ margin: "auto" }}>
							<h3>Loading...</h3>
						</Col>
						<Col xs={2} style={{ float: "left" }}>
							<Spinner animation='border' role='status'></Spinner>
						</Col>
					</Row>
				</Container>
			) : (
				<Container fluid className='contant-container'>
					<Row>
						<Col style={{ padding: "20px", paddingTop: "30px" }}>
							<Container>
								<Form>
									<Row>
										<Col xs={5}>
											<FormGroup>
												<Label>State</Label>
												<Input
													type='select'
													name='State'
													value={SelectedState}
													onChange={(e) => setSelectedState(e.target.value)}
													style={{ marginTop: "24px" }}>
													<option selected value>
														-- select an option --
													</option>
													{states()}
												</Input>

												{/* <Input
													style={{ marginTop: "24px" }}
													type='text'
													name='state'
													id='state'
													onChange={(e) => {
														setSelectedState(e.target.value);
													}}
													value={SelectedCity}
												/> */}
											</FormGroup>
										</Col>
										<Col xs={5}>
											<FormGroup>
												<Label>City</Label>
												<Input
													type='select'
													name='city'
													onChange={(e) => setSelectedCity(e.target.value)}
													value={SelectedCity}
													style={{ marginTop: "24px" }}>
													<option selected value>
														-- select an option --
													</option>
													{getCity(SelectedState)}
												</Input>

												{/* <Input
													style={{ marginTop: "24px" }}
													type='text'
													name='city'
													id='city'
													onChange={(e) => {
														setSelectedCity(e.target.value);
													}}
													value={SelectedCity}
												/> */}
											</FormGroup>
										</Col>
									</Row>
									<Row style={{ padding: "25px", marginTop: "24px" }}>
										<Col xs={4}>
											<FormGroup>
												<Label for='exampleSelect'>Crop</Label>
													<Input
														type='select'
														name='crop'
														value={SelectedCrop}
														onChange={(e) => setSelectedCrop(e.target.value)}
														style={{ marginTop: "24px" }}>
														<option value="">-- select a crop --</option>
														{CropYieldDropDownOption.crop_name.map(
														(crop, idx) => {
															return (
																<option key={idx} value={crop}>
																	{capitalizeFirstLetter(crop)}
																</option>
															);
														}
													)}
												</Input>
											</FormGroup>
										</Col>
										<Col xs={4}>
											<FormGroup>
												<Label for='exampleSelect'>Season</Label>
													<Input
														type='select'
														name='season'
														value={SelectedSeason}
														onChange={(e) => setSelectedSeason(e.target.value)}
														style={{ marginTop: "24px" }}>
														<option value="">-- select a season --</option>
														{SeasonName.season.map((season, idx) => {
														return (
															<option key={idx} value={season}>
																{capitalizeFirstLetter(season)}
															</option>
														);
													})}
												</Input>
											</FormGroup>
										</Col>
										<Col xs={4}>
											<FormGroup>
												<Label for='exampleCity'>Area</Label>
												<Input
													style={{ marginTop: "24px" }}
													type='text'
													name='area'
													id='Area'
													onChange={(e) => setArea(e.target.value)}
													value={Area}
												/>
											</FormGroup>
										</Col>
									</Row>
									<Row style={{ marginTop: "12px", marginBottom: "12px" }}>
										<Col style={{ padding: "20px", textAlign: "center" }}>
											<Button
												color='primary'
												onClick={(e) => {
													e.preventDefault();
													showResult();
												}}>
												Submit
											</Button>
										</Col>
									</Row>
									{isError ? <h4>{ErrorMessage}</h4> : null}
								</Form>
							</Container>
						</Col>
					</Row>
				</Container>
			)}
			{show ? (
				<>
					<Container fluid className='contant-container'>
						<Row>
							<Col>
								<h2
									style={{
										textAlign: "center",
										margin: "25px",
										marginBottom: "50px",
									}}>
									Season Yield Analysis
								</h2>
							</Col>
						</Row>
						<Row className='yield-result-row'>
							<Col>
								<Card
									body
									inverse
									style={{ backgroundColor: "#333", borderColor: "#333" }}>
									<CardTitle tag='h5'>Predicted Production</CardTitle>
									<CardText>{results.predProduction} / Quintal</CardText>
								</Card>
							</Col>
							<Col>
								<Card
									body
									inverse
									style={{ backgroundColor: "#333", borderColor: "#333" }}>
									<CardTitle tag='h5'>Predicted Yield</CardTitle>
									<CardText>{results.predYield} / Quintal/Hectare</CardText>
								</Card>
							</Col>
						</Row>
					</Container>

					<Container fluid className='contant-container'>
						<Row>
							<Col>
								<ReactApexChart
									options={YieldBarGraph.options}
									series={YieldBarGraph.series}
									type='bar'
									height={500}
								/>
							</Col>
						</Row>
					</Container>
					<Container fluid className='contant-container'>
						<Row>
							<Col>
								<ReactApexChart
									options={PieChart.options}
									series={PieChart.series}
									type='pie'
									height={500}
								/>
							</Col>
						</Row>
					</Container>
				</>
			) : null}
		</>
	);
};

export default YieldFinder;
