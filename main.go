package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

// 二分查找: arr必须是递增的
func binarySearch(arr []int, target int, l int, r int) int {
	if target > arr[r] || target < arr[l] {
		return -1
	}
	if l == r {
		if arr[l] == target {
			return l
		} else {
			return -1
		}
	}
	for l < r {
		middleIndex := (l + r) / 2
		if arr[middleIndex] == target {
			return middleIndex
		} else if arr[middleIndex] > target {
			return binarySearch(arr, target, l, middleIndex-1)
		} else {
			return binarySearch(arr, target, middleIndex+1, r)
		}
	}
	return -1
}
func main() {
	/* 打开保存uid的文件 */
	file, err := os.Open("../temp/reserve_uid.txt")
	if err != nil {
		fmt.Println("Error opening file:", err)
		return
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	var reserve_uid []int = make([]int, 0) // 最终长度:1678
	for scanner.Scan() {
		line := scanner.Text()
		uid, err_atoi := strconv.Atoi(line)
		if err_atoi != nil {
			fmt.Printf("Error Atoi: %s\n", line)
			return
		}
		reserve_uid = append(reserve_uid, uid)
	}

	if err := scanner.Err(); err != nil {
		fmt.Println("Error reading file:", err)
	}

	/****************************************************************/

	// 打开保存所有信息(7个字段 uid item_id author_id finish like device time)文件
	file, err = os.Open("../temp/temp.txt")
	if err != nil {
		fmt.Println("Error opening file:", err)
		return
	}
	defer file.Close()

	// 创建一个缓冲区
	reader := bufio.NewReader(file)

	// 设置读取进度的显示位置
	readPos := 0

	// 开始读取文件
	var uid_arr []int = make([]int, 0)
	var item_id_arr []int = make([]int, 0)
	var author_id_arr []int = make([]int, 0)
	var finish_arr []int = make([]int, 0)
	var like_arr []int = make([]int, 0)
	var device_arr []int = make([]int, 0)
	var time_arr []string = make([]string, 0)
	delete_cnt := 0
	start := time.Now()
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			break
		}
		line = line[:len(line)-1] // 去除换行符
		arr := strings.Split(line, " ")
		uid, _ := strconv.Atoi(arr[0])
		item_id, _ := strconv.Atoi(arr[1])
		author_id, _ := strconv.Atoi(arr[2])
		finish, _ := strconv.Atoi(arr[3])
		like, _ := strconv.Atoi(arr[4])
		device, _ := strconv.Atoi(arr[5])
		time := arr[6]

		pos := binarySearch(reserve_uid, uid, 0, len(reserve_uid)-1)
		if pos != -1 {
			// 需要保留的行
			uid_arr = append(uid_arr, uid)
			item_id_arr = append(item_id_arr, item_id)
			author_id_arr = append(author_id_arr, author_id)
			finish_arr = append(finish_arr, finish)
			like_arr = append(like_arr, like)
			device_arr = append(device_arr, device)
			time_arr = append(time_arr, time)
		} else {
			delete_cnt++
		}
		fmt.Printf("Reading: %.4f%%\r", (float64(readPos)/float64(275855530))*100.0) // 共计:275855530行
		readPos++
	}
	if len(uid_arr) == len(item_id_arr) && len(uid_arr) == len(finish_arr) && len(uid_arr) == len(author_id_arr) && len(uid_arr) == len(like_arr) && len(uid_arr) == len(device_arr) && len(uid_arr) == len(time_arr) {
		end := time.Since(start)
		fmt.Printf("\nReading Run Time: %.4f (hours)\n", end.Seconds()/3600)                   // 2.5h
		fmt.Printf("There are %d interaction left. %d been deleted", len(uid_arr), delete_cnt) // 保留:8,622,897; 删除:267,232,634
	} else {
		fmt.Println("Length Wrong")
		os.Exit(1)
	}

	/****************************************************************/

	/* 打开文件以进行写入 */
	// @@@@ 廖肇翊: uid item_id author_id time
	start = time.Now()
	file, err = os.Create("../temp/lzy.txt")
	if err != nil {
		fmt.Println("Error opening file:", err)
		return
	}
	defer file.Close()

	// 创建一个写入器
	writer := bufio.NewWriter(file)

	// 逐行写入文本
	line := "uid item_id author_id time\n"
	writer.WriteString(line)
	for i := 0; i < len(uid_arr); i++ {
		line := fmt.Sprintf("%d %d %d %s\n", uid_arr[i], item_id_arr[i], author_id_arr[i], time_arr[i])
		writer.WriteString(line)
	}

	// 刷新缓冲区
	writer.Flush()
	end := time.Since(start)
	fmt.Printf("\nWriting Run Time: %.4f (minutes)\n", end.Seconds()/60)

	// @@@@ Mine: uid item_id finish like device time
	start = time.Now()
	file, err = os.Create("../temp/reserve_samples.txt")
	if err != nil {
		fmt.Println("Error opening file:", err)
		return
	}
	defer file.Close()

	// 创建一个写入器
	writer = bufio.NewWriter(file)

	// 逐行写入文本
	for i := 0; i < len(uid_arr); i++ {
		line := fmt.Sprintf("%d %d %d %d %d %s\n", uid_arr[i], item_id_arr[i], finish_arr[i], like_arr[i], device_arr[i], time_arr[i])
		writer.WriteString(line)
	}

	// 刷新缓冲区
	writer.Flush()
	end = time.Since(start)
	fmt.Printf("\nWriting Run Time: %.4f (minutes)\n", end.Seconds()/60)

}
