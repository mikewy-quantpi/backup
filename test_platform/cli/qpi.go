package main

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"os/exec"
	"strings"

	"github.com/spf13/cobra"
	webview "github.com/webview/webview_go"
)

var HOST_URL = "http://" + os.Getenv("PICRYSTAL_HOST_URL")

func callEndPoint(url string) {
	response, err := http.Get(url)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	defer response.Body.Close()

	body, err := ioutil.ReadAll(response.Body)
	if err != nil {
		fmt.Println("Error reading response:", err)
		return
	}

	fmt.Println(string(body))
}

func curlResponse(output string) string {
	lines := strings.Split(output, "\n")

	if len(lines) > 3 {
		lines = lines[3:]
	}

	result := strings.Join(lines, "\n")
	return result
}

var rootCmd = &cobra.Command{
	Use:   "qpi",
	Short: "qpi controls PiCrystal AI test framework",
}

var buildCmd = &cobra.Command{
	Use:   "build",
	Short: "Build a machine learning use case Docker image",
	Run: func(cmd *cobra.Command, args []string) {
		useCaseFile, _ := cmd.Flags().GetString("use-case")
		requirementsFile, _ := cmd.Flags().GetString("requirements")
		imageName, _ := cmd.Flags().GetString("image-name")

		removeCommand := "rm .use_case.zip"
		defer exec.Command("bash", "-c", removeCommand).CombinedOutput()

		zipCommand := "zip " + ".use_case.zip" + " " + useCaseFile + " " + requirementsFile
		_, err := exec.Command("bash", "-c", zipCommand).CombinedOutput()
		if err != nil {
			fmt.Println("Error creating .use_case.zip file", err)
			return
		}

		apiEndPoint := "/create_use_case_image/"
		curlCommand := "curl -X POST " +
			"-F \"use_case_zip=@" + ".use_case.zip" + "\" " +
			HOST_URL + apiEndPoint + "?image_tag=" + imageName
		output, err := exec.Command("bash", "-c", curlCommand).CombinedOutput()
		if err != nil {
			fmt.Println("Error building use case Docker image", err)
			return
		}

		fmt.Println(curlResponse(string(output)))
	},
}

var statusCmd = &cobra.Command{
	Use:   "status",
	Short: "Get use case Docker image status",
	Run: func(cmd *cobra.Command, args []string) {
		imageName, _ := cmd.Flags().GetString("image-name")

		apiEndPoint := "/get_image_status/"
		url := fmt.Sprintf("%s%s%s", HOST_URL, apiEndPoint, imageName)
		callEndPoint(url)
	},
}

var executeCmd = &cobra.Command{
	Use:   "execute",
	Short: "Execute test suite",
	Run: func(cmd *cobra.Command, args []string) {
		imageName, _ := cmd.Flags().GetString("image-name")
		trustProfile, _ := cmd.Flags().GetString("trust-profile")

		apiEndPoint := "/execute_test_suite/"
		curlCommand := "curl -X POST " +
			"-F \"trust_profile=@" + trustProfile + "\" " +
			HOST_URL + apiEndPoint + "?image_name=" + imageName

		output, err := exec.Command("bash", "-c", curlCommand).CombinedOutput()
		if err != nil {
			fmt.Println("Error running test suite execution command", err)
			return
		}

		fmt.Println(curlResponse(string(output)))
	},
}

var resultCmd = &cobra.Command{
	Use:   "result",
	Short: "Get test result of a test suite",
	Run: func(cmd *cobra.Command, args []string) {
		imageName, _ := cmd.Flags().GetString("image-name")
		trustProfileName, _ := cmd.Flags().GetString("trust-profile-name")

		apiEndPoint := "/download_test_result/"
		url := fmt.Sprintf("%s%s?trust_profile_name=%s&image_name=%s", HOST_URL, apiEndPoint, trustProfileName, imageName)
		callEndPoint(url)
	},
}

var logsCmd = &cobra.Command{
	Use:   "logs",
	Short: "Check logs of test execution Docker container",
	Run: func(cmd *cobra.Command, args []string) {
		containerId, _ := cmd.Flags().GetString("container-id")

		apiEndPoint := "/check_container_logs/"
		url := fmt.Sprintf("%s%s?container_id=%s", HOST_URL, apiEndPoint, containerId)
		callEndPoint(url)
	},
}

var dashboardCmd = &cobra.Command{
	Use:   "dashboard",
	Short: "Open a dashboard that visualize the test result",
	Run: func(cmd *cobra.Command, args []string) {
		port := "8080"
		trustProfile, _ := cmd.Flags().GetString("trust-profile-name")
		if strings.Contains(strings.ToLower(trustProfile), "credit") {
			port = "9090"
		}

		w := webview.New(false)
		defer w.Destroy()
		w.SetTitle("QuantPi test result dashboard")
		w.SetSize(1024, 768, webview.HintNone)
		w.Navigate(fmt.Sprintf("%s:%s", HOST_URL, port))
		w.Run()
	},
}

func main() {
	buildCmd.Flags().String("use-case", "", "Path to use_case.py")
	buildCmd.Flags().String("requirements", "", "Path to requirements.txt")
	buildCmd.Flags().String("image-name", "", "Use case Docker image name")

	statusCmd.Flags().String("image-name", "", "Use case image name")

	executeCmd.Flags().String("image-name", "", "Use case image name")
	executeCmd.Flags().String("trust-profile", "", "Path to trust profile file")

	resultCmd.Flags().String("image-name", "", "Use case image name")
	resultCmd.Flags().String("trust-profile-name", "", "Trust profile name")

	logsCmd.Flags().String("container-id", "", "Test execution container id")

	dashboardCmd.Flags().String("image-name", "", "Use case image name")
	dashboardCmd.Flags().String("trust-profile-name", "", "Trust profile name")

	rootCmd.AddCommand(buildCmd)
	rootCmd.AddCommand(executeCmd)
	rootCmd.AddCommand(statusCmd)
	rootCmd.AddCommand(resultCmd)
	rootCmd.AddCommand(logsCmd)
	rootCmd.AddCommand(dashboardCmd)

	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}
