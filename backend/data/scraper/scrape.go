package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"

	"github.com/gocolly/colly"
)

// Product struct is to keep our web scraped information together
type Product struct {
	URL, Image, Name, Price string
}

func main() {
	fmt.Println("Hello web scraper")

	// This is how we initialize our web scraper
	c := colly.NewCollector(colly.AllowedDomains("www.scrapingcourse.com"))

	// Contains the products that we are attempting to scrape
	var products []Product

	// on HTML callbacks
	c.OnHTML(".product", func(e *colly.HTMLElement) {
		product := Product{}

		product.URL = e.ChildAttr("a", "href")
		product.Image = e.ChildAttr("img", "src")

		product.Name = e.ChildText(".product-name")
		product.Price = e.ChildText(".price")

		products = append(products, product)
	})

	c.OnScraped(func(r *colly.Response) {
		file, err := os.Create("products.csv")
		if err != nil {
			log.Fatalln("Failed to create output CSV file", err)
		}
		defer file.Close()

		// Initialize a file writer
		writer := csv.NewWriter(file)

		// Writing headers and records to the csv file to export
		headers := []string{
			"URL",
			"Image",
			"Name",
			"Price",
		}

		writer.Write(headers)

		for _, product := range products {
			record := []string{
				product.URL,
				product.Image,
				product.Name,
				product.Price,
			}

			writer.Write(record)
		}

		defer writer.Flush()
	})

	// Visits the website we are attempting to scrape
	c.Visit("https://www.scrapingcourse.com/ecommerce")

	fmt.Println("Hello web crawler")
}
