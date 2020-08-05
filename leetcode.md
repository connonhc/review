#### 目录

[toc]

#### 剑指 Offer

##### 03. 数组中的重复数字

找出数组中重复的数字。

在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字

**示例 1:**

```
输入：
[2, 3, 1, 0, 2, 5, 3]
输出：2 或 3 
```

**思路**

1. 将数字放入指定下标

   ```python
   class Solution:
       def findRepeatNumber(self, nums: List[int]) -> int:
           start = 0
           while True:
               if nums[start] == start:
                   start += 1
               else:
                   if nums[start] == nums[nums[start]]:
                       return nums[start]
                   else:
                       tmp = nums[start]
                       nums[start], nums[tmp] = nums[tmp], nums[start]
   ```

   

##### 04. 二维数组中的查找

在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

**示例 1:**

现有矩阵 matrix 如下：

```
[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
```

给定 target = 5，返回 true

给定 target = 20，返回 false

**思路**

1. 利用特性查找

   ```python
   class Solution:
       def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
           if not matrix: return False
           l, r = len(matrix) - 1, 0
           while l >= 0 and r <= len(matrix[0]) - 1:
               if matrix[l][r] > target:
                   l -= 1
               elif matrix[l][r] < target:
                   r += 1
               else:
                   return True
           return False
   ```

2. 利用 Python 成员运算符 in 遍历查找

   ```python
   class Solution:
       def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
           for i in matrix:
               if target in i:
                   return True
           return False
   ```

##### 05. 替换空格

请实现一个函数，把字符串 `s` 中的每个空格替换成"%20"。

**示例 1:**

```
输入：s = "We are happy."
输出："We%20are%20happy."
```

**思路**

1. 遍历生成新的列表

   ```python
   class Solution:
       def replaceSpace(self, s: str) -> str:
           res = []
           for i in s:
               if i == " ":
                   res.append("%20")
               else:
                   res.append(i)
           return "".join(res)
   ```

2. 使用 Python replace方法

   ```python
   class Solution:
       def replaceSpace(self, s: str) -> str:
           return s.replace(" ", "%20")
   ```

##### 06. 从尾到头打印链表

输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。

**示例 1:**

```
输入：head = [1,3,2]
输出：[2,3,1]
```

**思路**

1. 直接遍历

   ```python
   # Definition for singly-linked list.
   # class ListNode:
   #     def __init__(self, x):
   #         self.val = x
   #         self.next = None
   
   class Solution:
       def reversePrint(self, head: ListNode) -> List[int]:
           res = []
           while head:
               res.append(head.val)
               head = head.next
           # res.reverse()
           return res[::-1]
   ```

2. 递归

   ```python
   # Definition for singly-linked list.
   # class ListNode:
   #     def __init__(self, x):
   #         self.val = x
   #         self.next = None
   
   class Solution:
       def reversePrint(self, head: ListNode) -> List[int]:
           return self.reversePrint(head.next) + [head.val] if head else []
   ```


##### 07. 重建二叉树

输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。

**示例 1:**

例如，给出

```
前序遍历 preorder = [3,9,20,15,7]
中序遍历 inorder = [9,3,15,20,7]
```

返回如下的二叉树：

       3
      / \
     9  20
     /  \
    15   7


**思路**

1. 利用特性中序遍历左子树全在根节点前

   ```python
   # Definition for a binary tree node.
   # class TreeNode:
   #     def __init__(self, x):
   #         self.val = x
   #         self.left = None
   #         self.right = None
   
   class Solution:
       def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
           if len(preorder) == 0:
               return None
           head = TreeNode(preorder[0])
           left_node = inorder[0:inorder.index(preorder[0])]
           right_node = inorder[inorder.index(preorder[0])+1:]
           head.left = self.buildTree(preorder[1:len(left_node)+1], left_node)
           head.right = self.buildTree(preorder[len(left_node)+1:], right_node)
           return head
   ```

   ```python
   # Definition for a binary tree node.
   # class TreeNode:
   #     def __init__(self, x):
   #         self.val = x
   #         self.left = None
   #         self.right = None
   
   class Solution:
       def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
           self.dic, self.po = {}, preorder
           for i in range(len(inorder)):
               self.dic[inorder[i]] = i
           return self.recur(0, 0, len(inorder) - 1)
   
       def recur(self, pre_root, in_left, in_right):
           if in_left > in_right: return # 终止条件：中序遍历为空
           root = TreeNode(self.po[pre_root]) # 建立当前子树的根节点
           i = self.dic[self.po[pre_root]]    # 搜索根节点在中序遍历中的索引，从而可对根节点、左子树、右子树完成划分。
           root.left = self.recur(pre_root + 1, in_left, i - 1) # 开启左子树的下层递归
           root.right = self.recur(i - in_left + pre_root + 1, i + 1, in_right) # 开启右子树的下层递归
           return root # 返回根节点，作为上层递归的左（右）子节点
   ```

##### 10.1 斐波那契数列

写一个函数，输入 n ，求斐波那契（Fibonacci）数列的第 n 项。斐波那契数列的定义如下：

```
F(0) = 0,   F(1) = 1
F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
```

斐波那契数列由 0 和 1 开始，之后的斐波那契数就是由之前的两数相加而得出。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

**示例 1:**

```
输入：n = 2
输出：1
```

**示例 2:**

```
输入：n = 5
输出：5
```

**思路**

1. 递归(超时)

   ```python
   class Solution:
       def fib(self, n: int) -> int:
           return self.fib(n-1) + self.fib(n-2) if n > 1 else n
   ```

2. 记忆已经计算的数字

   ```python
   class Solution:
       def fib(self, n: int) -> int:
           res = [0, 1]
           for i in range(2, n+1):
               res.append(res[i-1] + res[i-2])
           return res[n] % 1000000007
   ```

3. 动态规划

   ```python
   class Solution:
       def fib(self, n: int) -> int:
           a, b = 0, 1
           for _ in range(n):
               a, b = b, a + b
           return a % 1000000007
   ```

##### 10. 2 青蛙跳台问题

一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1

**示例 1:**

```
输入：n = 2
输出：2
```

**示例 2:**

```
输入：n = 7
输出：21
```

**思路**

1. 斐波那契数列

   ```python
   class Solution:
       def numWays(self, n: int) -> int:
           res = [1, 1, 2]
           for i in range(3, n+1):
               res.append(res[i-1] + res[i-2])
           return res[n] % 1000000007
   ```

2. dp

   ```python
   class Solution:
       def numWays(self, n: int) -> int:
           a, b = 1, 1
           for _ in range(n):
               a, b = b, a + b
           return a % 1000000007
   ```

   ##### 12. 矩阵中的路径

   请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一格开始，每一步可以在矩阵中向左、右、上、下移动一格。如果一条路径经过了矩阵的某一格，那么该路径不能再次进入该格子。例如，在下面的3×4的矩阵中包含一条字符串“bfce”的路径（路径中的字母用加粗标出）。

   [["a","b","c","e"],
   ["s","f","c","s"],
   ["a","d","e","e"]]

   但矩阵中不包含字符串“abfb”的路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入这个格子

   **示例 1:**

   ```
   输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
   输出：true
   ```

   **示例 2:**

   ```
   输入：board = [["a","b"],["c","d"]], word = "abcd"
   输出：false
   ```

   **思路**

##### 15. 二进制中1的个数

请实现一个函数，输入一个整数，输出该数二进制表示中 1 的个数。例如，把 9 表示成二进制是 1001，有 2 位是 1。因此，如果输入 9，则该函数输出 2。

**示例 1:**

```
输入：00000000000000000000000000001011
输出：3
解释：输入的二进制串 00000000000000000000000000001011 中，共有三位为 '1'。
```

**示例 2:**

```
输入：00000000000000000000000010000000
输出：1
解释：输入的二进制串 00000000000000000000000010000000 中，共有一位为 '1'
```

**示例 3:**

```
输入：11111111111111111111111111111101
输出：31
解释：输入的二进制串 11111111111111111111111111111101 中，共有 31 位为 '1'
```

**思路**

1. 转二进制,计数

   ```python
   class Solution:
       def hammingWeight(self, n: int) -> int:
           count = 0
           while n:
               if n % 2:
                   count += 1
               n = n // 2
           return count
   ```


##### 28. 二叉树对称

请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。

例如，二叉树 [1,2,2,3,4,4,3] 是对称的。

       1
      / \
      2   2
     / \ / \
    3  4 4  3

但是下面这个 [1,2,2,null,3,null,3] 则不是镜像对称的:

    	1	
       / \
      2   2
       \   \
       3    3
**示例 1:**

```
输入：root = [1,2,2,3,4,4,3]
输出：true
```

**示例 2:**

```
输入：root = [1,2,2,null,3,null,3]
输出：false
```

**思路**

1. 递归

   ```python
   # Definition for a binary tree node.
   # class TreeNode:
   #     def __init__(self, x):
   #         self.val = x
   #         self.left = None
   #         self.right = None
   
   class Solution:
       def isSymmetric(self, root: TreeNode) -> bool:
           def recur(L, R):
               if not L and not R: return True
               if not L or not R or L.val != R.val: return False
               return recur(L.left, R.right) and recur(L.right, R.left)
           
           return recur(root.left, root.right) if root else True
   ```

##### 42. 最大连续整数和

输入一个整型数组，数组里有正数也有负数。数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。
要求时间复杂度为O(n)

**示例 1:**

```
输入: nums = [-2,1,-3,4,-1,2,1,-5,4]
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
```

**思路**

1. DP

   ```python
   class Solution:
       def maxSubArray(self, nums: List[int]) -> int:
           for i in range(1, len(nums)):
               nums[i] += max(nums[i - 1], 0)
           return max(nums)
   ```


##### 52. 两个链表的公共节点

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        node1, node2 = headA, headB
        
        while node1 != node2:
            node1 = node1.next if node1 else headB
            node2 = node2.next if node2 else headA

        return node1		
```

##### 57.1 两个和为s的数

输入一个**递增**排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可

**示例 1:**

```
输入：nums = [2,7,11,15], target = 9
输出：[2,7] 或者 [7,2]
```

**思路**

1. hash表遍历

   ```python
   class Solution:
       def twoSum(self, nums: List[int], target: int) -> List[int]:
           res = {}
           for i in nums:
               if i not in res:
                   res[target - i] = i
               else: return [i, target - i]
   ```

2. 双指针

   ```python
   class Solution:
       def twoSum(self, nums: List[int], target: int) -> List[int]:
           i, j =  0, len(nums) - 1
           while True:
               temp =  nums[i] + nums[j]
               if temp == target:
                   return [nums[i], nums[j]]
               elif temp < target:
                   i += 1
               else:
                   j -= 1
   ```

##### 57.2 为s的正整数序列

输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。

序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。

**示例 1:**

```
输入：target = 9
输出：[[2,3,4],[4,5]]
```

**思路**

1. 滑动窗口

   ```python
   class Solution:
       def findContinuousSequence(self, target: int) -> List[List[int]]:
           mid = target // 2 + 2
           l, r = 1, 2
           res = []
           while l < r and r < mid:
               t = (l + r) * ((r - l + 1) / 2)
               if t == target:
                   res.append([i for i in range(l, r + 1)])
                   r += 1
               elif t < target:
                   r += 1
               else:
                   l += 1
           return res
   ```

##### 29. 顺时针打印矩阵

输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。

**示例 1:**

```
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,3,6,9,8,7,4,5]
```

**思路**

1. 边界打印

   ```python
   class Solution:
       def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
           if not matrix: return []
           l, r, t, b = 0, len(matrix[0]) - 1, 0, len(matrix) - 1
           res = []
           while True:
               for i in range(l, r + 1): res.append(matrix[t][i])
               t += 1
               if t > b: break
               for i in range(t, b + 1): res.append(matrix[i][r])
               r -= 1
               if l > r: break
               for i in range(r, l - 1, -1): res.append(matrix[b][i])
               b -= 1
               if t > b: break
               for i in range(b, t - 1, -1): res.append(matrix[i][l])
               l += 1
               if l > r: break
           return res
   ```

##### 53.2 列表中缺失的数字

一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0～n-1之内。在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。

**示例 1:**

```
输入: [0,1,3]
输出: 2
```

**思路**

1. 遍历

2. 二分法

   ```python
   class Solution:
       def missingNumber(self, nums: List[int]) -> int:
           i, j = 0, len(nums) - 1
           while i <= j:
               m = (i + j) // 2
               if nums[m] == m: i = m + 1
               else: j = m - 1
           return i
   ```

##### 54. 二叉搜索树中的第K大节点

给定一棵二叉搜索树，请找出其中第k大的节点。

**示例**

```
输入: root = [3,1,4,null,2], k = 1
   3
  / \
 1   4
  \
   2
输出: 4
```

**思路**

1. 反向中序遍历

   ```python
   # Definition for a binary tree node.
   # class TreeNode:
   #     def __init__(self, x):
   #         self.val = x
   #         self.left = None
   #         self.right = None
   
   class Solution:
       def kthLargest(self, root: TreeNode, k: int) -> int:
           def dfs(root):
               if not root or self.res: return
               dfs(root.right)
               if self.k == 0: return
               self.k -= 1
               if self.k == 0: self.res = root.val
               dfs(root.left)
   
           self.k, self.res = k, None
           dfs(root)
           return self.res	
   ```


#### 题库

##### 001. 两数之和

```python
# 2020.08.03 01 simple
# dict 判断
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        res = {}
        for i, v in enumerate(nums):
            if v not in res:
                res[target - v] = i
            else: return [res[v], i]
```



##### 002. 两数相加

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# 2020.08.03 02 middle
# 注释
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        temp, res = 0, ListNode(0)
        p = res
        while temp or l1 or l2:
            if l1:
                temp += l1.val
                l1 = l1.next
            if l2:
                temp += l2.val
                l2 = l2.next
            p.next = ListNode(temp % 10)
            p = p.next
            temp = temp // 10
        return res.next
```



##### 003. 无重复最长子字符串

```python
# 2020.08.03 03 simple
# 滑动窗口
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        start, res, temp = -1, 0, {}
        for i, val in enumerate(s):
            if val in temp and temp[val] > start:
                start = temp[val]
                temp[val] = i
            else:
                temp[val] = i
                res = max(res, i - start)
        return res
```

##### 004. 寻找两个正序数组的中位数(TODO	

##### 005. 最长回文串

```python
# 2020.08.04 01 middle
# 中心拓展法 同时匹配左右
class Solution:
    def longestPalindrome(self, s: str) -> str:
        start, end = 0, 0
        for i in range(len(s)):
            left1, right1 = self.expandAroundCenter(s, i, i)
            left2, right2 = self.expandAroundCenter(s, i, i + 1)
            if right1 - left1 > end - start:
                start, end = left1, right1
            if right2 - left2 > end - start:
                start, end = left2, right2
        return s[start: end + 1]

    def expandAroundCenter(self, s, left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return left + 1, right - 1

```



##### 006. Z 字形变换

```python
# 2020.08.03 04 middle
# res[0] += c 
# res[1] += c 
# ...  
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows < 2: return s
        res = ["" for _ in range(numRows)]
        i, flag = 0, -1
        for c in s:
            res[i] += c
            if i == 0 or i == numRows - 1:
                flag = -flag
            i += flag
        return "".join(res)

        
```

##### 007. 整数反转

```python
# 2020.08.03 05 simple
# 注释
class Solution:
    def reverse(self, x: int) -> int:
        flag = False
        if x < 0:
            flag = True
            x = 0 - x
        x = int("".join(list(str(x))[::-1]))
        if flag: x = 0 - x
        return x if -2**31 <= x <= 2**31 - 1 else 0
    
        
```

##### 008. 字符串转换整数(TODO)

##### 009. 回文数

```python
# 2020.08.03 06 simple
# 反转 去除 '-' 号
class Solution:
    def isPalindrome(self, x: int) -> bool:
        return True if str(x).replace("-", "")[::-1] == str(x) else False
```

##### 010. 正则表达式匹配(TODO)

##### 11. 盛水最多的容器

```python
# 2020.08.04 02 middle
# 双指针
class Solution:
    def maxArea(self, height: List[int]) -> int:
        l, r, res = 0, len(height) - 1, 0
        while l < r:
            if height[l] < height[r]:
                res = max(res, height[l] * (r - l))
                l += 1
            else:
                res = max(res, height[r] * (r - l))
                r -= 1
        return res
```

##### 12. 整数转罗马数字

```python
# 2020.08.04 03 middle
# 从大到小减
class Solution:
    def intToRoman(self, num: int) -> str:
        digits = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), 
          (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
        roman_digits = []
        for value, symbol in digits:
            if num == 0: break
            count, num = divmod(num, value)
            roman_digits.append(symbol * count)
        return "".join(roman_digits)
```

##### 13. 罗马数字转整数

```python
# 2020.08.04 04 simple
# 遇到后面比前面大 就减后面的数
class Solution:
    def romanToInt(self, s: str) -> int:
        roman_dict = {
            "I": 1,
            "V": 5,
            "X": 10,
            "L": 50,
            "C": 100,
            "D": 500,
            "M": 1000
        }
        res = 0
        for i in range(len(s)):
            if i < len(s) - 1 and roman_dict[s[i]] < roman_dict[s[i+1]]:
                res -= roman_dict[s[i]]
            else:
                res += roman_dict[s[i]]
        return res
```



##### 14. 最长公共前缀

```python
# 2020.08.04 05 simple
# 遍历
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs: return ""
        res = strs[0]
        for i in strs[1:]:
            temp = ""
            for j, r in zip(i, res):
                if j == r:
                    temp += j
                else: break
            if temp == "":
                return ""
            res = temp
        return res
```

##### 15. 三数之和

```python
# 2020.08.04 06 middle
# 容易超时
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        nums.sort()
        ans = list()
        for first in range(n):
            if first > 0 and nums[first] == nums[first - 1]:
                continue
            third = n - 1
            target = -nums[first]
            for second in range(first + 1, n):
                if second > first + 1 and nums[second] == nums[second - 1]:
                    continue
                # target = -(nums[first] + nums[second])
                while second < third and nums[second] + nums[third] > target:
                    third -= 1
                if second == third:
                    break
                if nums[second] + nums[third] == target:
                    ans.append([nums[first], nums[second], nums[third]])
        return ans
```

##### 16. 组接近的三数和(TODO)

##### 17. 电话的字母组合

```python
# 2020.08.04 07 middle
# 递归
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits: return []
        nums_dict = {
        "2": ["a", "b", "c"],
        "3": ["d", "e", "f"],
        "4": ["g", "h", "i"],
        "5": ["j", "k", "l"],
        "6": ["m", "n", "o"],
        "7": ["p", "q", "r", "s"],
        "8": ["t", "u", "v"],
        "9": ["w", "x", "y", "z"],
        }

        def recur(nums):
            res = []
            if not nums: return [""]
            num = nums.pop(0)
            recur_item = recur(nums)
            for i in nums_dict[num]:
                for j in recur_item:
                    res.append(i + j)
            return res

        return recur(list(digits))
```

##### 18. 四数之和(TODO)

##### 19 删除链表的倒数第 N 个节点

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
# 2020.08.04 08 middle
# 双指针
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        temp_head = p = ListNode(0)
        p.next = head
        head = p
        while head:
            head = head.next
            if n != -1:
                n -= 1
            else: 
                p = p.next
        p.next = p.next.next if p.next.next else None
        return temp_head.next
```

##### 20. 有效的括号

```python
# 2020.08.04 09 simple
# 经典栈应用
class Solution:
    def isValid(self, s: str) -> bool:
        temp = {
        "(": ")",
        "{": "}",
        "[": "]"
        }
        res = []
        for i in s:
            if i not in temp and (not res or temp[res.pop()] != i):
                return False
            elif i in temp:
                res.append(i)
        return False if res else True
```

##### 21. 合并两个有序链表

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# 2020.08.04 10 simple
# 顺序遍历, 比较大小
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        head = p = ListNode(0)
        while l1 and l2:
            if l1.val < l2.val:
                p.next = ListNode(l1.val)
                l1 = l1.next
            else:
                p.next = ListNode(l2.val)
                l2 = l2.next
            p = p.next
        if l1: p.next = l1
        else: p.next = l2
        return head.next
```

##### 22. 括号生成

```python
# 2020.08.04 11 middle
# 不同位置插入 '()'
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        if n == 1: return ["()"]
        temp = self.generateParenthesis(n - 1)
        res = []
        for i in temp:
            for j in range(len(i)//2+1):
                res.append(i[:j] + "()" +i[j:])
        return list(set(res))
```

##### 23. 合并 K 个有序链表(TODO)

##### 24. 两两交换链表中的节点

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
# 2020.08.04 12 middle
# 递归
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        if not head or not head.next: return head
        temp, p = head, head.next.next
        head = head.next
        head.next = temp
        head.next.next = self.swapPairs(p)
        return head
```

##### 25. K 个一组翻转链表(TODO)

##### 26. 删除排序数组中的重复项

```python
# 2020.08.04 13 simple
# 遍历
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        l, r = 0, 1
        while r < len(nums):
            if nums[l] == nums[r]:
                nums.pop(r)
            else:
                l += 1
                r += 1
        return len(nums)
```

##### 27. 移除元素

```python
# 2020.08.04 14 simple
# 遍历
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        count = 0
        while count < len(nums):
            if nums[count] == val:
                nums.pop(count)
            else:
                count += 1
        return len(nums)
```

##### 28. 实现 strStr

```python
# 2020.08.04 15 simple
# 
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        if not needle: return 0
        count = 0
        start = 0
        left = 0
        while left < len(haystack):
            if haystack[left] == needle[count]:
                if not count:
                    start = left
                if count == len(needle) - 1:
                    return start
                count += 1
                left += 1
            elif not count:
                left += 1
            else:
                count = 0
                left = start + 1
        return -1
```

##### 29. 两数相除(TODO)

##### 30. 串联所有单词的子串(TODO)

##### 31. 下一个排列(TODO)

##### 32. 最长有效括号(TODO)



#### END

