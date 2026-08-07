enum TransactionType { debit, credit }

class TransactionModel {
  final String id;
  final double amount;
  final String merchant;
  final String category;
  final String bank;
  final TransactionType transactionType;
  final DateTime date;
  final String smsBody;
  final String smsHash;
  final DateTime createdAt;

  const TransactionModel({
    required this.id,
    required this.amount,
    required this.merchant,
    required this.category,
    required this.bank,
    required this.transactionType,
    required this.date,
    required this.smsBody,
    required this.smsHash,
    required this.createdAt,
  });

  // ─── Type Helpers ─────────────────────────────────────────────────────────

  bool get isDebit => transactionType == TransactionType.debit;
  bool get isCredit => transactionType == TransactionType.credit;

  // ─── fromMap (SQLite row → Model) ────────────────────────────────────────

  factory TransactionModel.fromMap(Map<String, dynamic> map) {
    return TransactionModel(
      id: map['id'] as String,
      amount: (map['amount'] as num).toDouble(),
      merchant: map['merchant'] as String,
      category: map['category'] as String,
      bank: map['bank'] as String? ?? '',
      transactionType: _parseType(map['transactionType'] as String?),
      date: DateTime.fromMillisecondsSinceEpoch(map['date'] as int),
      smsBody: map['smsBody'] as String? ?? '',
      smsHash: map['smsHash'] as String,
      createdAt: DateTime.fromMillisecondsSinceEpoch(map['createdAt'] as int),
    );
  }

  // ─── toMap (Model → SQLite row) ──────────────────────────────────────────

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'amount': amount,
      'merchant': merchant,
      'category': category,
      'bank': bank,
      'transactionType': transactionType.name, // 'debit' or 'credit'
      'date': date.millisecondsSinceEpoch,
      'smsBody': smsBody,
      'smsHash': smsHash,
      'createdAt': createdAt.millisecondsSinceEpoch,
    };
  }

  // ─── copyWith ────────────────────────────────────────────────────────────

  TransactionModel copyWith({
    String? id,
    double? amount,
    String? merchant,
    String? category,
    String? bank,
    TransactionType? transactionType,
    DateTime? date,
    String? smsBody,
    String? smsHash,
    DateTime? createdAt,
  }) {
    return TransactionModel(
      id: id ?? this.id,
      amount: amount ?? this.amount,
      merchant: merchant ?? this.merchant,
      category: category ?? this.category,
      bank: bank ?? this.bank,
      transactionType: transactionType ?? this.transactionType,
      date: date ?? this.date,
      smsBody: smsBody ?? this.smsBody,
      smsHash: smsHash ?? this.smsHash,
      createdAt: createdAt ?? this.createdAt,
    );
  }

  // ─── Equality ────────────────────────────────────────────────────────────

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is TransactionModel &&
          runtimeType == other.runtimeType &&
          id == other.id;

  @override
  int get hashCode => id.hashCode;

  // ─── Debug ───────────────────────────────────────────────────────────────

  @override
  String toString() {
    return 'Transaction(id: $id, amount: $amount, merchant: $merchant, '
        'category: $category, type: ${transactionType.name}, date: $date)';
  }

  // ─── Private Helpers ─────────────────────────────────────────────────────

  static TransactionType _parseType(String? raw) {
    if (raw == null) return TransactionType.debit;
    return raw.toLowerCase() == 'credit'
        ? TransactionType.credit
        : TransactionType.debit;
  }
}

// ─── Category Summary Model ───────────────────────────────────────────────────
// Used by dashboard and categories screen

class CategorySummary {
  final String category;
  final double totalAmount;
  final int transactionCount;

  const CategorySummary({
    required this.category,
    required this.totalAmount,
    required this.transactionCount,
  });

  factory CategorySummary.fromMap(Map<String, dynamic> map) {
    return CategorySummary(
      category: map['category'] as String,
      totalAmount: (map['totalAmount'] as num).toDouble(),
      transactionCount: map['transactionCount'] as int,
    );
  }
}

// ─── Dashboard Stats Model ────────────────────────────────────────────────────
// Single object passed from StatisticsService → dashboardProvider → UI

class DashboardStats {
  final double todaySpend;
  final double weekSpend;
  final double monthSpend;
  final String topCategory;
  final String topMerchant;
  final List<CategorySummary> categoryTotals;
  final List<TransactionModel> recentTransactions;

  const DashboardStats({
    required this.todaySpend,
    required this.weekSpend,
    required this.monthSpend,
    required this.topCategory,
    required this.topMerchant,
    required this.categoryTotals,
    required this.recentTransactions,
  });

  static DashboardStats empty() => const DashboardStats(
        todaySpend: 0,
        weekSpend: 0,
        monthSpend: 0,
        topCategory: '—',
        topMerchant: '—',
        categoryTotals: [],
        recentTransactions: [],
      );
}
