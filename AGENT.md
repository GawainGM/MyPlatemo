## 1. 基本调用方式

PlatEMO 的主入口是 `myPlatemo.m`。

```matlab
myPlatemo()
```

- 不带参数调用时：打开 PlatEMO GUI。
- 带参数调用时：直接在命令行运行指定算法和问题。

命令行调用的一般格式：

```matlab
myPlatemo('Name1',Value1,'Name2',Value2,...)
```

示例：

```matlab
myPlatemo('problem',@SOP_F1,'algorithm',@GA);
```

含义：使用算法 `GA` 求解问题 `SOP_F1`。

---

## 2. 常用参数

| 参数名 | 作用 | 典型取值 |
|---|---|---|
| `'algorithm'` | 指定优化算法 | `@GA`、`@NSGAII`、`@MOEAD` 等 |
| `'problem'` | 指定测试问题/优化问题 | `@SOP_F1`、`@DTLZ2`、`@WFG1` 等 |
| `'N'` | 种群规模 | 正整数，如 `50`、`100` |
| `'M'` | 目标数 | 正整数，如 `2`、`3`、`5` |
| `'D'` | 决策变量维度 | 正整数，如 `6`、`100` |
| `'maxFE'` | 最大函数评价次数 | 正整数，如 `10000`、`20000` |
| `'maxRuntime'` | 最大运行时间，单位秒 | 正数，如 `10` |
| `'save'` | 保存的种群数量/保存开关 | 整数，如 `5`、`6` |
| `'run'` | 当前独立运行编号 | 整数，如 `1`、`2` |
| `'metName'` | 需要计算的性能指标 | `{'IGD','HV'}` |
| `'outputFcn'` | 每代/每次迭代后的输出函数 | 函数句柄，如 `@DefaultOutput` |

---

## 3. 算法和问题的指定方式

### 3.1 直接指定算法

```matlab
myPlatemo('algorithm',@GA,'problem',@SOP_F1);
```

### 3.2 给算法传入参数

如果算法自身需要参数，使用 cell 数组：

```matlab
myPlatemo('algorithm',{@GA,1,30,1,30});
```

其中：

```matlab
{@GA,p1,p2,...}
```

表示调用算法 `GA`，并传入算法参数 `p1,p2,...`。

### 3.3 直接指定问题

```matlab
myPlatemo('problem',@SOP_F1,'algorithm',@GA);
```

### 3.4 给问题传入参数

如果问题自身需要参数，也使用 cell 数组：

```matlab
myPlatemo('problem',{@WFG1,20});
```

其中：

```matlab
{@WFG1,p1,p2,...}
```

表示调用问题 `WFG1`，并传入问题参数 `p1,p2,...`。

---

## 4. 设置种群规模、目标数、维度和终止条件

### 4.1 设置种群规模 `N`

```matlab
myPlatemo('algorithm',@GA,'problem',@SOP_F1,'N',50);
```

含义：使用 `GA` 求解 `SOP_F1`，种群规模为 `50`。

### 4.2 设置目标数 `M`

```matlab
myPlatemo('algorithm',@NSGAII,'problem',@DTLZ2,'M',5);
```

含义：使用 `NSGAII` 求解 5 目标的 `DTLZ2` 问题。

### 4.3 设置决策变量维度 `D`

```matlab
myPlatemo('algorithm',@GA,'problem',@SOP_F1,'D',100);
```

含义：使用 `GA` 求解 100 维的 `SOP_F1` 问题。

### 4.4 设置最大函数评价次数 `maxFE`

```matlab
myPlatemo('algorithm',@GA,'problem',@SOP_F1,'maxFE',20000);
```

含义：最大函数评价次数为 `20000`。

### 4.5 设置最大运行时间 `maxRuntime`

```matlab
myPlatemo('algorithm',@GA,'problem',@SOP_F1,'maxRuntime',10);
```

含义：最多运行 `10` 秒。

注意：`maxRuntime` 和 `maxFE` 都可作为终止条件。实际使用中，常根据实验需求选择其一或同时设置。

---

## 5. 保存实验结果

使用 `'save'` 参数保存结果：

```matlab
myPlatemo('save',Value);
```

保存后的数据通常位于：

```text
myPlatemo/Data/算法名/
```

文件名通常包含：

```text
alg_pro_M_D_run.mat
```

其中：

- `alg`：算法名；
- `pro`：问题名；
- `M`：目标数；
- `D`：决策变量维度；
- `run`：运行编号。

### 多次独立运行并保存

```matlab
parfor i = 1 : 100
    myPlatemo('algorithm',@NSGAII,...
            'problem',@DTLZ2,...
            'save',5,...
            'run',i);
end
```

含义：并行运行 100 次，每次使用不同的 `run` 编号保存结果。

---

## 6. 计算性能指标

通过 `'metName'` 指定需要计算的指标。

示例：

```matlab
myPlatemo('algorithm',@NSGAII,...
        'problem',@DTLZ2,...
        'save',6,...
        'metName',{'IGD','HV'});
```

含义：运行后计算 `IGD` 和 `HV` 指标，并保存相关结果。

常见指标包括：

- `IGD`
- `HV`
- `GD`
- `Spread`
- `Spacing`

具体可用指标以 `Metrics` 文件夹中的 `.m` 文件为准。

---

## 7. 手动调用指标计算

若已有结果 `result`，可手动计算指标。例如：

```matlab
pro = DTLZ2();
pro.CalMetric('IGD',result{end});
```

含义：对最后一代/最终结果计算 `IGD`。

---